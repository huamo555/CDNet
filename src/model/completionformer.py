"""
    CompletionFormer
    ======================================================================

    CompletionFormer implementation
"""

from .nlspn_module import NLSPN
from .backbone import Backbone
import torch
import torch.nn as nn

class CompletionFormer(nn.Module):
    def __init__(self, args):
        super(CompletionFormer, self).__init__()

        self.args = args
        self.prop_time = self.args.prop_time
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        self.backbone = Backbone(args, mode='rgbd')

        if self.prop_time > 0:
            self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                    self.args.prop_kernel)

    def forward(self, rgb, dep):

        dep = dep.unsqueeze(1)
        rgb = rgb.float()  # 将输入的 RGB 图像转换为 FloatTensor
        dep = dep.float()  # 将输入的 Depth 图像转换为 FloatTensor

        pred_init = self.backbone(rgb, dep)
        return pred_init
