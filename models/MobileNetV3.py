import torch
from torch import nn
import torch.nn.functional as F
class InvertedResidualConfig:
    def __init__(self, input_c, kernel, expanded_c, out_c, use_se, activation, stride, width_multi):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels, width_multi):
        return _make_divisible(channels * width_multi, 8)

class InvertedResidual(nn.Module):
    def __init__(self, cnf, norm_layer):
        super(InvertedResidual, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c, cnf.expanded_c, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel, stride=cnf.stride, groups=cnf.expanded_c, norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity))
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
    
        return result

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, activation_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x

def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch