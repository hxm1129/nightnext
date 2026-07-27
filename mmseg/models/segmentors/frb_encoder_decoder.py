from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from .frbnet_utils import FIINet


@SEGMENTORS.register_module()
class FRBEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 number_K=10,
                 lamda=0.1):
        super(FRBEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.frb_net = FIINet(number_K=number_K, lamda=lamda)

    def extract_feat(self, img):
        img = self.frb_net(img)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x