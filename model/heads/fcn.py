from torch import nn
import torch


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]
        super().__init__(*layers)


if __name__ == '__main__':
    model = FCNHead(48, 3)
    print(model)
    x = torch.randn((1, 48, 256, 256))
    y = model(x)
    print(x.shape)
    print(y.shape)