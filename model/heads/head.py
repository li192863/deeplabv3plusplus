from .fcn import FCNHead


def get_head(in_channels=48, num_classes=3):
    """
    获取头部
    @param num_classes: 种类数
    @return:
    """
    return FCNHead(in_channels=in_channels, channels=num_classes)