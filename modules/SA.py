# -*- coding: utf-8 -*-
# @Date     : 2025/12/13
# @Author   : Zhou
# @File     : SA.py
# description : This is the code for the Spatial Attention (SA) module.

import torch
import torch.nn as nn
class SA(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out) * x
    
if __name__ == '__main__':
    input=torch.randn(64,64,32,32)
    block = SA(kernel_size=7)
    output=block(input)
    print(output.shape)