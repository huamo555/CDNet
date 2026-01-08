"""
Depth Filler Network.

Author: Hongjie Fang.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dense import DenseBlock
from .duc import DenseUpsamplingConvolution
from .MobileNetV3 import InvertedResidualConfig, InvertedResidual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class DFNet(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=16, L=5, k=12, use_DUC=True, **kwargs):
        super(DFNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.L = L
        self.k = k
        self.use_DUC = use_DUC

        # 初始化所有层
        self.ff = nn.Sequential(
            nn.Conv2d(68, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        ).to(device)
        self.first = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        ).to(device)
        self.dense1s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        ).to(device)
        
        # 初始化 InvertedResidual 层 skip
        cnf1 = InvertedResidualConfig(input_c=self.hidden_channels, kernel=3, expanded_c=self.hidden_channels, out_c=self.hidden_channels, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        self.block1 = InvertedResidual(cnf1, norm_layer).to(device)
        
        # 继续初始化其他层...
        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        ).to(device)
        cnf2 = InvertedResidualConfig(input_c=self.hidden_channels , kernel=3, expanded_c=self.hidden_channels, out_c=self.hidden_channels, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.block2 = InvertedResidual(cnf2, norm_layer).to(device)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels * 2, kernel_size=3, stride=2, padding=1).to(device)
        
        self.dense2s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels * 2),
            nn.ReLU(True)
        ).to(device)
        cnf3 = InvertedResidualConfig(input_c=self.hidden_channels * 2, kernel=3, expanded_c=self.hidden_channels * 2, out_c=self.hidden_channels * 2, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.block3 = InvertedResidual(cnf3, norm_layer).to(device)
        
        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels* 2, self.hidden_channels* 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels* 2),
            nn.ReLU(True)
        ).to(device)
        cnf4 = InvertedResidualConfig(input_c=self.hidden_channels* 2, kernel=3, expanded_c=self.hidden_channels* 2, out_c=self.hidden_channels* 2, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.block4 = InvertedResidual(cnf4, norm_layer).to(device)
        self.conv4 = nn.Conv2d(in_channels=self.hidden_channels* 2, out_channels=self.hidden_channels * 4, kernel_size=3, stride=2, padding=1).to(device)
        
        self.dense3s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels* 4, self.hidden_channels* 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels* 4),
            nn.ReLU(True)
        ).to(device)
        cnf5 = InvertedResidualConfig(input_c=self.hidden_channels* 4, kernel=3, expanded_c=self.hidden_channels* 4, out_c=self.hidden_channels* 4, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.block5 = InvertedResidual(cnf5, norm_layer).to(device)
        
        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels* 4, self.hidden_channels* 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels* 4),
            nn.ReLU(True)
        ).to(device)
        cnf6 = InvertedResidualConfig(input_c=self.hidden_channels* 4, kernel=3, expanded_c=self.hidden_channels* 4, out_c=self.hidden_channels* 4, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.block6 = InvertedResidual(cnf6, norm_layer).to(device)
        self.conv6 = nn.Conv2d(in_channels=self.hidden_channels* 4, out_channels=self.hidden_channels * 8, kernel_size=3, stride=2, padding=1).to(device)
        
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels* 8, self.hidden_channels* 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels* 8),
            nn.ReLU(True)
        ).to(device)
        cnf7 = InvertedResidualConfig(input_c=self.hidden_channels* 8, kernel=3, expanded_c=self.hidden_channels* 8, out_c=self.hidden_channels* 8, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.block7 = InvertedResidual(cnf7, norm_layer).to(device)
        
        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels* 8, self.hidden_channels* 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels* 8),
            nn.ReLU(True)
        ).to(device)
        upcnf1 = InvertedResidualConfig(input_c=self.hidden_channels* 8, kernel=3, expanded_c=self.hidden_channels* 8, out_c=self.hidden_channels* 8, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.upblock1 = InvertedResidual(upcnf1, norm_layer).to(device)
        self.upsample_block1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_channels* 8,
                out_channels=self.hidden_channels* 4,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.hidden_channels* 4),
            nn.ReLU(True)
        ).to(device)
        
        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 8, self.hidden_channels* 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels * 4),
            nn.ReLU(True)
        ).to(device)
        upcnf2 = InvertedResidualConfig(input_c=self.hidden_channels* 4, kernel=3, expanded_c=self.hidden_channels* 4, out_c=self.hidden_channels* 4, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.upblock2 = InvertedResidual(upcnf2, norm_layer).to(device)
        self.upsample_block2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_channels* 4,
                out_channels=self.hidden_channels* 2,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.hidden_channels* 2),
            nn.ReLU(True)
        ).to(device)
        
        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 4, self.hidden_channels* 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels* 2),
            nn.ReLU(True)
        ).to(device)
        upcnf3 = InvertedResidualConfig(input_c=self.hidden_channels* 2, kernel=3, expanded_c=self.hidden_channels* 2, out_c=self.hidden_channels* 2, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.upblock3 = InvertedResidual(upcnf3, norm_layer).to(device)
        self.upsample_block3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_channels* 2,
                out_channels=self.hidden_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        ).to(device)
        
        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        ).to(device)
        upcnf4 = InvertedResidualConfig(input_c=self.hidden_channels, kernel=3, expanded_c=self.hidden_channels, out_c=self.hidden_channels, use_se=False, activation="RE", stride=1, width_multi=1.0)
        self.upblock4 = InvertedResidual(upcnf4, norm_layer).to(device)
        self.upsample_block4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        ).to(device)
        
        self.final =nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 800, kernel_size=1, stride=1)
        ).to(device)
        self.obj_mask = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 2, kernel_size=1, stride=1)
        ).to(device)

    def forward(self, rgb, depth):
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)

        rgb = rgb.float()
        depth = depth.float()
        h = self.first(torch.cat((rgb, depth), dim=1))
        
        # dense1: skip
        h_d1s = self.dense1s_conv1(h) #  [1, 64, 360, 640]
        h_d1s = self.block1(h_d1s)
        
        # dense1: normal
        h = self.dense1_conv1(h)  #  [1, 64, 360, 640]
        h = self.block2(h)
        h = self.conv2(h)         #  [1, 128, 180, 360]
        
        # dense2: skip
        h_d2s = self.dense2s_conv1(h)  
        h_d2s = self.block3(h_d2s)
        
        # dense2: normal
        h = self.dense2_conv1(h)
        h = self.block4(h)
        h = self.conv4(h)
        
        # dense3: skip
        h_d3s = self.dense3s_conv1(h)
        h_d3s = self.block5(h_d3s)
        
        # dense3: normal
        h = self.dense3_conv1(h)
        h = self.block6(h)
        h = self.conv6(h)
        
        # dense4
        h = self.dense4_conv1(h)
        h = self.block7(h)
        
        # updense1
        h = self.updense1_conv(h)
        h = self.upblock1(h)
        h = self.upsample_block1(h)
        
        # updense2
        h = torch.cat((h, h_d3s), dim=1)
        h = self.updense2_conv(h)
        h = self.upblock2(h)
        h = self.upsample_block2(h)
        
        # updense3
        h = torch.cat((h, h_d2s), dim=1)
        h = self.updense3_conv(h)
        h = self.upblock3(h)
        h = self.upsample_block3(h)
        
        # updense4
        h = torch.cat((h, h_d1s), dim=1)
        h = self.updense4_conv(h)
        h = self.upblock4(h)
        h = self.upsample_block4(h)
        
        # final
        w = self.final(h)
        obj = self.obj_mask(h)

        return w, obj

# 将模型移到设备上
model = DFNet().to(device)