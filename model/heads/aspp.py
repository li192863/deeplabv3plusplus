from typing import Sequence

import torch
from torch import nn

from torch.nn import functional as F


class ASPPConv(nn.Sequential):
    """ ASPP卷积分支 """
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ ASPP池化分支 """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)  # 要求batch_size >= 2


class ASPP(nn.Module):
    """ ASPP模块 """
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        # 分支1
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        )
        # 分支2/3/4
        rates = tuple(atrous_rates)  # [6, 12, 18]
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        # 分支5
        modules.append(ASPPPooling(in_channels, out_channels))
        # 合并所有分支
        self.convs = nn.ModuleList(modules)
        # 1x1卷积压缩
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))  # 要求batch_size >= 2
        res = torch.cat(_res, dim=1)
        return self.project(res)


if __name__ == '__main__':
    aspp = ASPP(in_channels=160, atrous_rates=[6, 12, 16], out_channels=256)
    print(aspp)
    x = torch.randn((2, 160, 8, 8))
    print(aspp(x).shape)  # [2, 256, 8, 8]