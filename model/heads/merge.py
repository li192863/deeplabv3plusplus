import torch
from torch import nn

from .aspp import ASPP


class MergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, atrous_rates=[6, 12, 16], final=False):
        super(MergeBlock, self).__init__()
        self.in_channels_high = in_channels
        self.in_channels_low = in_channels // 2
        self.out_channels = out_channels
        assert self.in_channels_low == self.out_channels, f'in_channels {self.in_channels_low} must == out_channels {self.out_channels}'
        # high -> aspp + up
        if bilinear:
            self.hnet = nn.Sequential(
                ASPP(in_channels=self.in_channels_high, atrous_rates=atrous_rates, out_channels=self.in_channels_low),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(self.in_channels_low, self.in_channels_low, kernel_size=1),
                nn.BatchNorm2d(self.in_channels_low),
                nn.ReLU(inplace=True)
            )
        else:
            self.hnet = nn.Sequential(
                ASPP(in_channels=self.in_channels_high, atrous_rates=atrous_rates, out_channels=self.in_channels_low),
                nn.ConvTranspose2d(self.in_channels_low, self.in_channels_low, kernel_size=2, stride=2),
                nn.BatchNorm2d(self.in_channels_low),
                nn.ReLU(inplace=True)
            )
        # low -> shorcut
        if not final:
            self.lnet = nn.Sequential(
                nn.Conv2d(self.in_channels_low, self.in_channels_low, kernel_size=1),
                nn.BatchNorm2d(self.in_channels_low),
                nn.ReLU(inplace=True)
            )
        else:
            if bilinear:
                self.lnet = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(self.in_channels_high, self.in_channels_low, kernel_size=1),
                    nn.BatchNorm2d(self.in_channels_low),
                    nn.ReLU(inplace=True)

                )
            else:
                self.lnet = nn.Sequential(
                    nn.ConvTranspose2d(self.in_channels_high, self.in_channels_low, kernel_size=2, stride=2),
                    nn.Conv2d(self.in_channels_low, self.in_channels_low, kernel_size=1),
                    nn.BatchNorm2d(self.in_channels_low),
                    nn.ReLU(inplace=True)
                )
        # (low + high) -> cat_conv
        self.mnet = nn.Sequential(
            nn.Conv2d(self.in_channels_high, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )


    def forward(self, low, high):
        high = self.hnet(high)
        low = self.lnet(low)
        merged = torch.cat((low, high), dim=1)
        return self.mnet(merged)


class Merge(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3, in_channels4, out_channels, bilinear=True, atrous_rates=[6, 12, 16]):
        super(Merge, self).__init__()
        self.mnet1 = MergeBlock(in_channels4, in_channels3, bilinear=bilinear, atrous_rates=atrous_rates, final=False)
        self.mnet2 = MergeBlock(in_channels3, in_channels2, bilinear=bilinear, atrous_rates=atrous_rates, final=False)
        self.mnet3 = MergeBlock(in_channels2, in_channels1, bilinear=bilinear, atrous_rates=atrous_rates, final=False)
        self.mnet4 = MergeBlock(in_channels1, out_channels, bilinear=bilinear, atrous_rates=atrous_rates, final=True)
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)

            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, out1, out2, out3, out4):
        out3 = self.mnet1(out3, out4)
        out2 = self.mnet2(out2, out3)
        out1 = self.mnet3(out1, out2)
        out = self.mnet4(x, out1)
        return self.up(out)


def get_merge(bilinear=True):
    """
    获取解码器
    @return: 模型
    """
    merge = Merge(96, 192, 384, 768, 48, bilinear=bilinear)
    return merge

