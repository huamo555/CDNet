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
    # assert (kernel % 2) == 1, \
    #     'only odd kernel is supported but kernel = {}'.format(kernel)

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
    """
    Depth Filler Network (DFNet).
    """
    def __init__(self, in_channels = 4, hidden_channels = 16, L = 5, k = 12, use_DUC = True, **kwargs):
        super(DFNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.L = L
        self.k = k
        self.use_DUC = use_DUC
        # self.masks = nn.Parameter(torch.Tensor( 1, 1, 720, 640))

        self.ff = nn.Sequential(
            nn.Conv2d(68, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # First
        self.first = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: skip
        self.dense1s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        
        # Dense1: normal
        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        
        # Dense2: skip
        self.dense2s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        

        # Dense2: normal
        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )

        # Dense3: skip
        self.dense3s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )

        # Dense3: normal
        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )

        # Dense4
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )

        # DUC upsample 1
        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )

        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )

        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2 , self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )

        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 800, kernel_size = 1, stride = 1)
        )
        self.obj_mask = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 2, kernel_size = 1, stride = 1)
        )


    def forward(self, rgb, depth):
        # 720 x 640 (rgb, depth) -> 360 x 640 (h)
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)

        rgb = rgb.float()
        depth = depth.float()
        h = self.first(torch.cat((rgb, depth), dim = 1))
        # dense1: 360 x 640 (h, depth1) -> 180 x 320 (h, depth2)
        depth1 = F.interpolate(depth, scale_factor = 0.5, mode = "nearest")
        
        # dense1: skip
        h_d1s = self.dense1s_conv1(h)
        cnf1 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        block1 = InvertedResidual(cnf1, norm_layer)
        block1 = block1.to(device)
        h_d1s = block1(h_d1s)
        
        
        # dense1: normal
        h = self.dense1_conv1(h)
        cnf2 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        block2 = InvertedResidual(cnf2, norm_layer)
        block2 = block2.to(device)
        h = block2(h)
        conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        conv2 = conv2.to(device)
        h = conv2(h)  
        
        # dense2: 180 x 320 (h, depth2) -> 90 x 160 (h, depth3)
        depth2 = F.interpolate(depth1, scale_factor = 0.5, mode = "nearest")
        # dense2: skip
        
        
        h_d2s = self.dense2s_conv1(h)
        cnf3 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        block3 = InvertedResidual(cnf3, norm_layer)
        block3 = block3.to(device)
        h_d2s = block3(h_d2s)
        

        # dense2: normal
        h = self.dense2_conv1(h)
        cnf4 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        block4 = InvertedResidual(cnf4, norm_layer)
        block4 = block4.to(device)
        h = block4(h)
        conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        conv4 = conv4.to(device)
        h = conv4(h)
        
        
        # dense3: 90 x 160 (h, depth3) -> 45 x 80 (h, depth4)
        depth3 = F.interpolate(depth2, scale_factor = 0.5, mode = "nearest")
        
        # dense3: skip
        h_d3s = self.dense3s_conv1(h)
        cnf5 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        block5 = InvertedResidual(cnf5, norm_layer)
        block5 = block5.to(device)
        h_d3s = block5(h_d3s)
        
        
        # dense3: normal
        h = self.dense3_conv1(h)
        cnf6 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        block6 = InvertedResidual(cnf6, norm_layer)
        block6 = block6.to(device)
        h = block6(h)
        conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        conv6 = conv6.to(device)
        h = conv6(h)
        

        # dense4: 45 x 80
        depth4 = F.interpolate(depth3, scale_factor = 0.5, mode = "nearest")
        h = self.dense4_conv1(h)
        cnf7 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        block7 = InvertedResidual(cnf7, norm_layer)
        block7 = block7.to(device)
        h = block7(h)
        


        # updense1: 45 x 80 -> 90 x 160
        h = self.updense1_conv(h)
        upcnf1 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        upblock1 = InvertedResidual(upcnf1, norm_layer)
        upblock1 = upblock1.to(device)
        h = upblock1(h)
        upsample_block1 = nn.Sequential(
          nn.ConvTranspose2d(
          in_channels=64,
          out_channels=64,
          kernel_size=2,
          stride=2,
          padding=0,
          output_padding=0,
          bias=False  
          ),
          nn.BatchNorm2d(64),
          nn.ReLU(True)
        )
        upsample_block1 = upsample_block1.to(device)
        h = upsample_block1(h)


        # updense2: 90 x 160 -> 180 x 320
        h = torch.cat((h, h_d3s), dim = 1)
        h = self.updense2_conv(h)
        upcnf2 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        upblock2 = InvertedResidual(upcnf2, norm_layer)
        upblock2 = upblock2.to(device)
        h = upblock2(h)
        upsample_block2 = nn.Sequential(
          nn.ConvTranspose2d(
          in_channels=64,
          out_channels=64,
          kernel_size=2,
          stride=2,
          padding=0,
          output_padding=0,
          bias=False  
          ),
          nn.BatchNorm2d(64),
          nn.ReLU(True)
        )
        upsample_block2 = upsample_block2.to(device)
        h = upsample_block2(h)

        # updense3: 180 x 320 -> 360 x 640
        h = torch.cat((h, h_d2s), dim = 1)
        h = self.updense3_conv(h)
        upcnf3 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        upblock3 = InvertedResidual(upcnf3, norm_layer)
        upblock3 = upblock3.to(device)
        h = upblock3(h)
        upsample_block3 = nn.Sequential(
          nn.ConvTranspose2d(
          in_channels=64,
          out_channels=64,
          kernel_size=2,
          stride=2,
          padding=0,
          output_padding=0,
          bias=False  
          ),
          nn.BatchNorm2d(64),
          nn.ReLU(True)
        )
        upsample_block3 = upsample_block3.to(device)
        h = upsample_block3(h)

        # updense4: 360 x 640 -> 720 x 640
        h = torch.cat((h, h_d1s), dim = 1)
        h = self.updense4_conv(h)
        upcnf4 = InvertedResidualConfig(input_c=64, kernel=3, expanded_c=64, out_c=64, use_se=False, activation="RE", stride=1, width_multi=1.0)
        norm_layer = nn.BatchNorm2d
        upblock4 = InvertedResidual(upcnf4, norm_layer)
        upblock4 = upblock4.to(device)
        h = upblock4(h)
        upsample_block4 = nn.Sequential(
          nn.ConvTranspose2d(
          in_channels=64,
          out_channels=64,
          kernel_size=2,
          stride=2,
          padding=0,
          output_padding=0,
          bias=False  
          ),
          nn.BatchNorm2d(64),
          nn.ReLU(True)
        )
        upsample_block4 = upsample_block4.to(device)
        h = upsample_block4(h)
        
        # final
        w = self.final(h)

        obj = self.obj_mask(h)

        # return rearrange(h, 'n 1 h w -> n h w')
        
        return w , obj


