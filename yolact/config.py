from detectron2.config import CfgNode as CN


def add_yolact_config(cfg):
    cfg.MODEL.YOLACT = CN()
    # mask parameters:
    cfg.MODEL.YOLACT.NUM_MASKS = 32
    cfg.MODEL.YOLACT.DISCARD_MASK_AREA = 5*5

    # loss parameters:
    cfg.MODEL.YOLACT.SEM_SEG_ALPHA = 1.0
    cfg.MODEL.YOLACT.MASK_ALPHA = 6.125
    cfg.MODEL.YOLACT.MASK_REWEIGHT = 1.0
    cfg.MODEL.YOLACT.MASKIOU_ALPHA = 25
