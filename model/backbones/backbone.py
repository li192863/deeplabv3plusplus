from torchvision.models.feature_extraction import create_feature_extractor

from .swinv2 import swinv2_tiny_window16_256


def get_backbone(pretrained=False):
    """
    获取主干网络
    @return: (模型, low_level_channels, x_channels)
    """
    # model2 = SwinTransformerV2(
    #     window_size=16,
    #     img_size=(512, 512),
    #     embed_dim=128,
    #     depths=(2, 2, 18, 2),
    #     num_heads=(4, 8, 16, 32)
    # )
    model = swinv2_tiny_window16_256(pretrained=pretrained)
    # model2.load_state_dict(torch.load('../model.safetensors'), strict=False)
    return_nodes = {
        'patch_embed': 'x',  # [64, 64, 96]
        'layers.0': 'out1',  # [64, 64, 96]
        'layers.1': 'out2',  # [32, 32, 192]
        'layers.2': 'out3',  # [16, 16, 384]
        'layers.3': 'out4',  # [8, 8, 768]
        # 'norm': 'x',       # [8, 8, 768]
    }
    return create_feature_extractor(model, return_nodes=return_nodes)
