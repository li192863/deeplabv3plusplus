import torch
import transforms


class SegmentationPresetTrain:
    def __init__(
            self,
            *,
            base_size=(256, 256),
            crop_size=(256, 256),
            hflip_prob=0.5,
            mean=(0.723, 0.485, 0.608),
            std=(0.293, 0.377, 0.333),
    ):
        # 随机缩放
        trans = [transforms.RandomResize(int(0.5 * sum(base_size) / 2.0), int(2.0 * sum(base_size) / 2.0))]
        # 随机水平翻转
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                # 随机剪裁
                transforms.RandomCrop(int(sum(crop_size) / 2.0)),
                transforms.PILToTensor(),
                transforms.ToDtype(torch.float, scale=True),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = transforms.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
            self,
            *,
            base_size=(256, 256),
            crop_size=None,
            mean=(0.723, 0.485, 0.608),
            std=(0.293, 0.377, 0.333),
    ):
        self.transforms = transforms.Compose(
            [
                # 随机缩放
                transforms.RandomResize(min_size=int(sum(base_size) / 2.0), max_size=int(sum(base_size) / 2.0)),
                transforms.PILToTensor(),
                transforms.ToDtype(torch.float, scale=True),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)
