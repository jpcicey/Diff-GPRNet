# -*- coding: utf-8 -*-
# @Date     : 2025/12/13
# @Author   : Zhou
# @File     : DWT.py
# description : This is the code for the network structure of the residual wavelet downsampling module.

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# from torch.cuda.amp import autocast

class Down_wt(nn.Module):
    """Downsampling with wavelet transform"""
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_ch*4, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.residual_conv(x)
        with torch.amp.autocast('cuda', enabled=False):
            yL, yH = self.wt(x.float())
        yH = yH[0]
        y_HL, y_LH, y_HH = yH[:, :, 0, :, :], yH[:, :, 1, :, :], yH[:, :, 2, :, :]
        yH_cat = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        out = self.main_conv(yH_cat)
        out = out + residual
        return out

if __name__ == "__main__":
    model = Down_wt(1, 1)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (in millions): {total_params / 1e6:.4f} M")
