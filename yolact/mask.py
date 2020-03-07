import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec


class ProtoNet(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        in_channels = input_shape.channels
        # in_stride = input_shape.stride

        num_masks   = cfg.MODEL.YOLACT.NUM_MASKS

        self.proto = nn.Sequential([
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_masks, 1),
            nn.ReLU(inplace=True)
        ])
    
    def forward(self, x):
        x = self.proto(x)


class MaskIouNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        in_channels = 1
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES

        self.maskiou = nn.Sequential([
            nn.Conv2d(in_channels, 8, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
            nn.ReLU(inplace=True)
        ])
    
    def forward(self, x):
        x = self.maskiou(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)
        return x
