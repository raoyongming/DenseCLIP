import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class IdentityHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(IdentityHead, self).__init__(
            input_transform=None, **kwargs)
        self.conv_seg = None

    def forward(self, inputs):
        return inputs