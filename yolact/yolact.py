import logging
import math
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, BoxMode
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY

from .head import YolactHead
from .mask import ProtoNet, MaskIouNet
from .utils import (crop, mask_iou, permute_to_N_HWA_K, 
    permute_all_cls_and_box_to_N_HWA_K_and_concat)

__all__ = ["Yolact"]


@META_ARCH_REGISTRY.register()
class Yolact(nn.Module):
    """
    Implement Yolact.
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        self.device                   = torch.device(cfg.MODEL.DEVICE)
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES
        # Mask parameters:
        self.discard_mask_area        = cfg.MODEL.YOLACT.DISCARD_MASK_AREA
        self.num_masks                = cfg.MODEL.YOLACT.NUM_MASKS
        # Loss parameters:
        self.sem_seg_alpha            = cfg.MODEL.YOLACT.SEM_SEG_ALPHA
        self.mask_alpha               = cfg.MODEL.YOLACT.MASK_ALPHA
        self.mask_reweight            = cfg.MODEL.YOLACT.MASK_REWEIGHT
        self.maskiou_alpha            = cfg.MODEL.YOLACT.MASKIOU_ALPHA
        self.focal_loss_alpha         = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        # retinanet_resnet_fpn_backbone
        self.backbone = build_backbone(cfg)
        # dict[str->ShapeSpec]
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        # base retinanet add mask coefficient branch 
        self.head = YolactHead(cfg, feature_shapes)
        # which layer output of backbone to protonet. see offical yolact's cfg.proto_src.
        # default is `res2`, but this is `res3`
        self.protonet = ProtoNet(cfg, feature_shapes[0])
        # to mask scoring
        self.maskiou_net = MaskIouNet(cfg)
        # semantic segmentation to help training
        self.semantic_seg_conv = nn.Conv2d(feature_shapes[0].channels, self.num_classes, 1)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # ["p3", "p4", "p5", "p6", "p7"] features
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, mask_coef = self.head(features)
        proto_mask = self.protonet(features[0])
        sem_seg = self.semantic_seg_conv(features[0])
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas, gt_matched_idxs = self.get_ground_truth(
                anchors, gt_instances)
            det_loss = self.detection_loss(
                gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)
            mask_loss, maskiou_data = self.lincomb_mask_loss(
                gt_classes, mask_coef, proto_mask, gt_instances, gt_matched_idxs)
            sem_seg_loss = self.semantic_segmentation_loss(sem_seg, gt_instances)
            losses = {}
            losses.update(det_loss)
            losses.update(mask_loss)
            losses.update(sem_seg_loss)
            if maskiou_data is not None:
                maskiou_loss = self.maskiou_loss(*maskiou_data)
                losses.update(maskiou_loss)
            return losses
        else:
            results = self.inference(box_cls, box_delta, anchors, mask_coef, proto_mask,
                                     images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = self.postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def detection_loss(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`Yolact.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`YolactHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        # one-hot
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def lincomb_mask_loss(self, gt_classes, mask_coef, proto_mask, gt_instances, gt_matched_idxs):
        """
        Args:
            gt_classes: shapes are (N, R). See :meth:`Yolact.get_ground_truth`.
            mask_coef (list[Tensor]): lvl tensors, each has shape (N, Ax#masks, Hi, Wi).
                See :meth:`YolactHead.forward`.
            proto_mask (Tensor): shapes are (N, #masks, M, M).
            gt_instances (list[Instances]): a list of N `Instances`s.
            gt_matched_idxs (list[Tensor[int64]]): each element is a vector of length R, 
                where gt_matched_idxs[i] is a matched ground-truth index in [0, #objects)
        Return:
            loss_mask [dict]: mask loss scalar.
            maskiou_data (list[inputs, targets, classes]): the input of maskiou_net.
        """
        mask_size = proto_mask.size()[-2:]
        mask_area = mask_size[0] * mask_size[1]
        # shape: (N, M, M, #masks)
        proto_mask = proto_mask.permute(0, 2, 3, 1).contiguous()

        gt_masks = []
        gt_boxes = []
        gt_boxes_area = [] # for normalize weight
        gt_masks_area = [] # for discard_mask_area
        mask_weights = []
        with torch.no_grad():
            for i, instance_per_image in enumerate(gt_instances):
                gt_mask = instance_per_image.gt_masks.to(device=proto_mask.device).tensor
                gt_mask = gt_mask.permute(1,2,0).contiguous()
                gt_mask = F.interpolate(gt_mask, mask_size, mode="bilinear", align_corners=False)
                # gt_mask: shape (M, M, #objects)
                gt_mask = gt_mask.gt(0.5).float()
                gt_masks.append(gt_mask)
                gt_masks_area.append(gt_mask.sum(dim=(0, 1)))
                # mask weights
                gt_foreground_norm = gt_mask / (gt_mask.sum(dim=(0,1), keepdim=True) + 0.0001)
                gt_background_norm = (1-gt_mask) / ((1-gt_mask).sum(dim=(0,1), keepdim=True) + 0.0001)
                mask_weight = (gt_foreground_norm * self.mask_reweight + gt_background_norm) * mask_area
                mask_weights.append(mask_weight)
                # :class:`Boxes` shape (#objects, 4)
                # convert to relative coordinate to crop mask
                gt_box = BoxMode.convert(instance_per_image.gt_boxes, BoxMode.XYXY_ABS, BoxMode.XYXY_REL)
                gt_boxes.append(gt_box.tensor)
                # area(#objects)
                gt_boxes_area.append(gt_box.area())
      
        # convert to aligned with gt_classes
        mask_coef = [permute_to_N_HWA_K(x, self.num_masks) for x in mask_coef]
        # Tensor shape (N, R, #masks)
        mask_coef = cat(mask_coef, dim=1)

        mask_loss = 0
        maskiou_inputs = []
        maskiou_targets = []
        maskiou_classes = []
        # combine mask_coef and proto_mask to generate pred_mask of each image 
        # and calculate loss
        for i in range(len(gt_instances)):
            # gt_class
            gt_class = gt_classes[i]
            # -1: ignore, #num_classes: background
            foreground_idxs = (gt_class >= 0) & (gt_class != self.num_classes)
            pred_coef = mask_coef[i, foreground_idxs]
            # matrix multiply get shape (M, M, #pos)
            pred_mask = F.sigmoid(proto_mask[i] @ pred_coef.t())

            # matched ground truth objects' idx
            gt_matched_idx = gt_matched_idxs[i][foreground_idxs]
            # generate gt_masks
            gt_box = gt_boxes[i][gt_matched_idx]
            gt_mask = gt_masks[i][gt_matched_idx]
            # crop mask using gt_box
            pred_mask = crop(pred_mask, gt_box)
            
            pre_loss = F.binary_cross_entropy(
                torch.clamp(pred_mask, 0, 1), gt_mask, reduction='none')
            # mask_proto_reweight_mask_loss: foreground and background has different weights
            pre_loss = pre_loss * mask_weights[i][:, :, gt_matched_idx]
            # mask_proto_normalize_emulate_roi_pooling: 
            # Normalize the mask loss to emulate roi pooling's affect on loss.
            pre_loss = pre_loss.sum(dim=(0, 1)) * (mask_area / gt_boxes_area[i])

            mask_loss += pre_loss.sum()

            # cfg.use_maskiou
            select = gt_masks_area[i] > self.discard_mask_area
            if select.sum() > 0:
                pred_mask = pred_mask[:, :, select]
                gt_mask = gt_mask[:, :, select]
                gt_class = gt_class[select]
                # maskiou net input: (N, 1, H, W)
                maskiou_input = pred_mask.permute(2, 0, 1).contiguous().unsqueeze(1)
                pred_mask = pred_mask.gt(0.5).float()
                # maskiou net target: (N)             
                maskiou_target = mask_iou(pred_mask, gt_mask)
                maskiou_inputs.append(maskiou_input)
                maskiou_targets.append(maskiou_target)
                maskiou_classes.append(gt_class)

        losses = {"loss_mask": mask_loss / mask_area * self.mask_alpha}

        if len(maskiou_targets) == 0:
            return losses, None
        else:
            # all images have same size masks
            # so the tensor are shape (N*I, 1, H, W)
            maskiou_targets = torch.cat(maskiou_targets)
            maskiou_classes = torch.cat(maskiou_classes)
            maskiou_inputs = torch.cat(maskiou_inputs)
            return losses, (maskiou_inputs, maskiou_targets, maskiou_classes)

    def mask_iou_loss(self, pred_masks, targets, classes):
        """
        Args:
            pred_masks (Tensor[N, 1, M, M])： some pos pred_masks in a batch.
            targets (Tensor[N]): truth iou between pred_masks and its gt_masks.
            classes (Tensor[N]): gt classes of each pred_masks.
        """
        # shape (N, C)
        pred_iou = self.maskiou_net(pred_masks)
        classes = classes[:, None]
        pred_iou = torch.gather(pred_iou, dim=1, index=classes).view(-1)
        iou_loss = F.smooth_l1_loss(pred_iou, targets, reduction='sum')
        return {"loss_iou": iou_loss * self.maskiou_alpha}

    def semantic_segmentation_loss(self, sem_seg, gt_instances):
        """
        Args:
            sem_seg (Tensor[N, C, H, W]): semantic segmentation pred.
            gt_instances (list[Instances]):
        """
        mask_size = sem_seg.size()[-2:]
        gt_segment = torch.zeros_like(sem_seg, requires_grad=False)
        with torch.no_grad():
            for idx, instances in enumerate(gt_instances):
                gt_mask = instances.gt_masks.tensor
                gt_class = instances.gt_classes
                gt_mask = F.interpolate(gt_mask.unsqueeze(0), mask_size,
                    mode="bilinear", align_corners=False).squeeze(0)
                gt_mask = gt_mask.gt(0.5).float()
                # Construct Semantic Segmentation
                for obj_idx, gt_mask_per_obj in enumerate(gt_mask):
                    gt_segment[idx, gt_class[obj_idx]] = torch.max(
                        gt_segment[idx, gt_class[obj_idx]], gt_mask_per_obj)

        sem_loss = F.binary_cross_entropy_with_logits(
            sem_seg, gt_segment, reduction='mean')
        return {"loss_sem_seg" : sem_loss * self.sem_seg_alpha}
  
    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
            gt_matched_idxs_all (list[Tensor[int64]]): each element is a vector of length N, 
                where gt_matched_idxs_all[i] is a matched ground-truth index in [0, M)
        """
        gt_classes = []
        gt_anchors_deltas = []
        gt_matched_idxs_all = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_per_image.tensor, matched_gt_boxes.tensor
                )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
            gt_matched_idxs_all.append(gt_matched_idxs)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas), gt_matched_idxs_all

    def inference(self, box_cls, box_delta, anchors, mask_coef, proto_masks, image_sizes):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(image_sizes)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        mask_coef = [permute_to_N_HWA_K(x, self.num_masks) for x in mask_coef]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4 or #masks)
        
        proto_masks = proto_masks.permute(0, 2, 3, 1).contiguous()
        # Tensor (N, M, M, #masks)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = image_sizes[img_idx]
            proto_mask_per_image = proto_masks[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            mask_coef_per_image = [mask_coef_per_level[img_idx] for mask_coef_per_level in mask_coef]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image, mask_coef_per_image,
                proto_mask_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, mask_coef, proto_mask, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            mask_coef (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, #masks)
            proto_mask (Tensor): size (M, M, #masks)
            image_size (tuple(H, W)): a tuple of the image height and width.
        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        mask_coef_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, mask_coef_i, anchors_i in zip(
            box_cls, box_delta, mask_coef, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs] # (N,4)
            anchors_i = anchors_i[anchor_idxs]
            mask_coef_i = mask_coef_i[anchor_idxs] # (N, #masks)
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
            mask_coef_all.append(mask_coef_i)

        boxes_all, scores_all, class_idxs_all, mask_coef_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all, mask_coef_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]

        pred_masks = F.sigmoid(proto_mask @ mask_coef_all[keep].t())
        # note: pred_masks shape (M, M, #keep)
        pred_masks = crop(pred_masks, boxes_all[keep])
        # shape (#keep, M, M)
        pred_masks = pred_masks.permute(2, 0, 1).contiguous()
        # mask_iou to rescore mask
        if self.rescore_mask:
            pred_maskiou = self.maskiou_net(pred_masks.unsqueeze(1))
            pred_maskiou = torch.gather(
                pred_maskiou, dim=1, index=class_idxs_all[keep].unsqueeze(1)).squeeze(1)
            result.scores = scores_all[keep] * pred_maskiou

        pred_masks = F.interpolate(pred_masks.unsqueeze(0), image_size, 
            mode="bilinear", align_corners=False).squeeze(0)
        result.pred_masks = pred_masks.gt_(0.5)
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def postprocess(self, results, output_height, output_width):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.

        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.

        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.

        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        results = Instances((output_height, output_width), **results.get_fields())
        output_boxes = results.pred_boxes
        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)
        results = results[output_boxes.nonempty()]
        return results
