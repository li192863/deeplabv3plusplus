import torch
from torch import nn

from .backbones import get_backbone
from .heads import get_head, get_merge


class MyNet(nn.Module):
    """ DeepLabV3+模型 """
    def __init__(self, bilinear=True, num_classes=3):
        super(MyNet, self).__init__()
        self.backbone = get_backbone(pretrained=False)
        self.neck = get_merge(bilinear=bilinear)
        self.head = get_head(in_channels=48, num_classes=num_classes)

    def forward(self, x):
        # backbone
        out = self.backbone(x)
        # HWC -> CHW
        x = out['x'].permute(0, 3, 1, 2).contiguous()
        out1 = out['out1'].permute(0, 3, 1, 2).contiguous()
        out2 = out['out2'].permute(0, 3, 1, 2).contiguous()
        out3 = out['out3'].permute(0, 3, 1, 2).contiguous()
        out4 = out['out4'].permute(0, 3, 1, 2).contiguous()
        # neck
        out = self.neck(x, out1, out2, out3, out4)
        # head
        out = self.head(out)
        return out


if __name__ == '__main__':
    model = MyNet(bilinear=False, num_classes=4)
    print(model)
    x = torch.randn((2, 3, 256, 256))
    print(model(x).shape)
