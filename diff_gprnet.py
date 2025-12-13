# -*- coding: utf-8 -*-
# @Date     : 2025/12/13
# @Author   : Zhou
# @File     : diff_gprnet.py
# description : This is the code for the network structure of Diff-GPRNet.

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.SA import SA
from modules.ECA import ECA
from modules.DWT import Down_wt


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class RMA(nn.Module):
    """
    Residual Mixed Attention module
    Can be applied to the skip connection part of U-Net:
    1. Adaptive handling of channel number mismatch (automatic matching via 1×1 convolution)
    2. Support for ECA+SA fusion
    """
    def __init__(self, in_channels, out_channels):
        super(RMA, self).__init__()
        # Double convolution block
        self.dconv = DoubleConv(in_channels, out_channels)

        # Attention modules
        self.eca = ECA(out_channels)
        self.sa = SA()
        # self.alpha = nn.Parameter(torch.tensor(0.5))  # Adaptive fusion coefficient

        # Channel matching shortcut
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        # Residual connection
        residual = self.shortcut(x)
        out_d = self.dconv(x)
        residual2 = self.shortcut(out_d)

        # Mixed attention
        out_e = self.eca(out_d)
        out_s = self.sa(out_e)

        out = residual + out_s + residual2
        return self.ReLU(out)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            Down_wt(in_channels, out_channels),
            RMA(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling using ConvTranspose2d then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # Upsampling using transposed convolution, halving the number of channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Number of input channels after concatenation: (in_channels//2 + out_channels)
        self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        # x1: Features from the decoder (to be upsampled)
        # x2: Skip connection features from the encoder
        x1 = self.up(x1)
        
        # Handle size mismatch (padding)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        # Concatenate features
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class Diff_GPRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Diff_GPRNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Bottleneck
        self.bottleneck = RMA(1024, 1024)

        # Decoder (upsampling + RMA)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024 channels

        # Bottleneck processing
        x5 = self.bottleneck(x5)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    begin = time.time()
    model = Diff_GPRNet(in_channels=1, out_channels=1)
    input_data = torch.randn(1, 1, 256, 256)
    output = model(input_data)
    end = time.time()

    print("Time taken:", end - begin)
    print("Input shape:", input_data.shape)
    print("Output shape:", output.shape)
    print("Model parameters (in millions):", sum(p.numel() for p in model.parameters()) / 1e6, "M")