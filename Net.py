import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvLayer(nn.Module):
  def __init__(self, in_c, out_c, kernel_size):
    super().__init__()
    pad = int(np.floor(kernel_size/2))
    self.conv = nn.Conv2d(in_c, out_c, kernel_size = kernel_size, stride = 1, padding = pad)
  def forward(self, x):
    return self.conv(x)

class Bottleneck(nn.Module):
  def __init__(self, in_c, out_c, kernel_size = 3, stride=1):
    super().__init__()
    self.in_c = in_c
    self.out_c = out_c
    self.kernel_size = kernel_size
    self.identity_block = nn.Sequential(
        ConvLayer(in_c, out_c//4, kernel_size=1),
        nn.InstanceNorm2d(out_c//4),
        nn.ReLU(),
        ConvLayer(out_c//4, out_c//4, kernel_size),
        nn.InstanceNorm2d(out_c//4),
        nn.ReLU(),
        ConvLayer(out_c//4, out_c, kernel_size=1),
        nn.InstanceNorm2d(out_c),
        nn.ReLU()
    )
    self.shortcut = nn.Sequential(
        ConvLayer(in_c, out_c, 1),
        nn.InstanceNorm2d(out_c),
    )


  def forward(self, x):
    out = self.identity_block(x)
    if self.in_c == self.out_c:
      residual = x
    else:
      residual = self.shortcut(x)
    out =+ residual
    out = F.relu(out)
    return out

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = F.interpolate(out, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        out = self.norm(out)
        out = F.relu(out)
        return out

def upsample(scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear')


class HRNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1_1 = Bottleneck(3, 16)

    self.layer2_1 = Bottleneck(16, 32)
    self.downsample2_1 = nn.Conv2d(16, 32, kernel_size=3, stride = 2, padding=1)

    self.layer3_1 = Bottleneck(32, 32)
    self.layer3_2 = Bottleneck(32, 32)
    self.downsample3_1 = nn.Conv2d(32, 32, kernel_size=3, stride = 2, padding=1)
    self.downsample3_2 = nn.Conv2d(32, 32, kernel_size=3, stride = 4, padding=1)
    self.downsample3_3 = nn.Conv2d(32, 32, kernel_size=3, stride = 2, padding=1)

    self.layer4_1 = Bottleneck(64, 64)
    self.layer5_1 = Bottleneck(192, 64)
    self.layer6_1 = Bottleneck(64, 32)
    self.layer7_1 = Bottleneck(32, 16)
    self.layer8_1 = nn.Conv2d(16, 3, kernel_size=3, stride = 1, padding=1)

  def forward(self, x):
    map1_1 = self.layer1_1(x)

    map2_1 = self.layer2_1(map1_1)
    map2_2 = self.downsample2_1(map1_1)

    map3_1 = torch.cat((self.layer3_1(map2_1), upsample(2)(map2_2)), 1)
    map3_2 = torch.cat((self.downsample3_1(map2_1), self.layer3_2(map2_2)), 1)
    map3_3 = torch.cat((self.downsample3_2(map2_1), self.downsample3_3(map2_2)), 1)

    map4_1 = torch.cat((self.layer4_1(map3_1), upsample(2)(map3_2), upsample(4)(map3_3)), 1)

    out = self.layer5_1(map4_1)
    out = self.layer6_1(out)
    out = self.layer7_1(out)
    out = self.layer8_1(out)

    return out