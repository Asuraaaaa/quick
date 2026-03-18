from __future__ import annotations

import torch
from torch import nn


class TransformerBackbone(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.d_model = d_model
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = self.encoder(x)
        x = self.norm(x)
        # print(x.shape)
        return x

import torch
import torch.nn as nn
import torchvision.models as tvm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# =========================================================
# 1. 通用工具：修改首层输入通道数
# =========================================================
def replace_first_conv_in_sequential(module: nn.Module, in_chans: int):
    """
    在 module 内递归找到第一个 Conv2d，并替换为新的输入通道数。
    适用于 torchvision 的大多数 CNN 主干。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=in_chans,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups if in_chans % child.groups == 0 else 1,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode,
            )

            # 如果原来是 3 通道，尽量做一个合理初始化
            with torch.no_grad():
                if child.weight.shape[1] == 3 and in_chans == 2:
                    # 用前两个通道初始化
                    new_conv.weight[:, :2] = child.weight[:, :2]
                elif child.weight.shape[1] == 1 and in_chans > 1:
                    new_conv.weight[:] = child.weight.repeat(1, in_chans, 1, 1) / in_chans
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)

            setattr(module, name, new_conv)
            return True

        replaced = replace_first_conv_in_sequential(child, in_chans)
        if replaced:
            return True
    return False


# =========================================================
# 2. 通用 Backbone Wrapper
#    输入 [B, 2, 129, 63] -> 输出 [B, 256]
# =========================================================
class CNNBackbone256(nn.Module):
    def __init__(self, features: nn.Module, feat_dim: int, out_dim: int = 256):
        super().__init__()
        self.features = features
        self.proj = nn.Linear(feat_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x: [B, 2, 129, 63]
        return: [B, 256]
        """
        x = self.features(x)           # [B, C, H', W']
        x = torch.flatten(x, 1)        # [B, C*H'*W']
        x = self.proj(x)               # [B, 256]
        return self.dropout(x)


# =========================================================
# 3. MobileNetV2
# =========================================================
def build_mobilenet_v2_backbone(pretrained=False, out_dim=256):
    weights = tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = tvm.mobilenet_v2(weights=weights)

    replace_first_conv_in_sequential(model.features, in_chans=2)

    features = model.features
    feat_dim = model.last_channel   # 1280
    return CNNBackbone256(features, 12800, out_dim)


# =========================================================
# 4. MobileNetV3-Small
# =========================================================
def build_mobilenet_v3_small_backbone(pretrained=False, out_dim=256):
    weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = tvm.mobilenet_v3_small(weights=weights)

    replace_first_conv_in_sequential(model.features, in_chans=2)

    features = model.features
    feat_dim = model.classifier[0].in_features   # 通常是 576
    return CNNBackbone256(features, 5760, out_dim)


# =========================================================
# 5. ShuffleNetV2-x1.5
# =========================================================
class ShuffleNetV2Backbone256(nn.Module):
    def __init__(self, model: nn.Module, out_dim=256):
        super().__init__()
        self.conv1 = model.conv1
        self.maxpool = model.maxpool
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.conv5 = model.conv5
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(10240, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x


def build_shufflenet_v2_x1_5_backbone(pretrained=False, out_dim=256):
    weights = tvm.ShuffleNet_V2_X1_5_Weights.DEFAULT if pretrained else None
    model = tvm.shufflenet_v2_x1_5(weights=weights)

    # ShuffleNet 的首层是 model.conv1[0]
    old_conv = model.conv1[0]
    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    with torch.no_grad():
        if old_conv.weight.shape[1] == 3:
            new_conv.weight[:, :2] = old_conv.weight[:, :2]
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
    model.conv1[0] = new_conv

    return ShuffleNetV2Backbone256(model, out_dim)


# =========================================================
# 6. EfficientNet B0
# =========================================================
def build_efficientnet_b0_backbone(pretrained=False, out_dim=256):
    weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)

    replace_first_conv_in_sequential(model.features, in_chans=2)

    features = model.features
    feat_dim = model.classifier[1].in_features   # 通常是 1280
    return CNNBackbone256(features, 12800, out_dim)


# =========================================================
# 7. ConvNeXt Atto (依赖 timm)
# =========================================================
class TimmBackbone256(nn.Module):
    def __init__(self, model: nn.Module, out_dim=256):
        super().__init__()
        self.model = model
        self.proj = nn.Linear(1280, out_dim)

    def forward(self, x):
        # timm 中 forward_features 返回 [B, C, H, W] 或 [B, C]
        x: torch.Tensor = self.model.forward_features(x)

        x = x.flatten(start_dim=1)

        x = self.proj(x)
        return x

class TimmBackbonemobilevit256(nn.Module):
    def __init__(self, model: nn.Module, out_dim=256):
        super().__init__()
        self.model = model
        self.proj = nn.Linear(3840, out_dim)

    def forward(self, x):
        # timm 中 forward_features 返回 [B, C, H, W] 或 [B, C]
        x: torch.Tensor = self.model.forward_features(x)

        x = x.flatten(start_dim=1)

        x = self.proj(x)
        return x


def build_convnextv2_atto_backbone(pretrained=False, out_dim=256):
    if not HAS_TIMM:
        raise ImportError(
            "构造 ConvNeXt Atto 需要 timm，请先安装: pip install timm"
        )

    # timm 中的 convnext_atto
    model = timm.create_model(
        'convnextv2_atto',
        pretrained=pretrained,
        in_chans=2,
        num_classes=0,   # 去掉原分类头
        global_pool=''   # 自己做 pooling
    )
    return TimmBackbone256(model, out_dim)


def build_mobilevit_xs_backbone(pretrained=False, out_dim=256):
    if not HAS_TIMM:
        raise ImportError(
            "构造 MobileViT XS 需要 timm，请先安装: pip install timm"
        )

    model = timm.create_model(
        'mobilevit_xs',
        pretrained=pretrained,
        in_chans=2,
        num_classes=0,   # 去掉原分类头
        global_pool=''   # 自己做 pooling
    )
    return TimmBackbonemobilevit256(model, out_dim)


# =========================================================
# 8. 统一工厂函数
# =========================================================
def build_backbone(name: str, pretrained=False, out_dim=256):
    name = name.lower()

    if name == 'mobilenet_v2':
        return build_mobilenet_v2_backbone(pretrained, out_dim)

    elif name == 'mobilenet_v3_small':
        return build_mobilenet_v3_small_backbone(pretrained, out_dim)

    elif name == 'shufflenet_v2_x1_5':
        return build_shufflenet_v2_x1_5_backbone(pretrained, out_dim)

    elif name == 'convnextv2_atto':
        return build_convnextv2_atto_backbone(pretrained, out_dim)

    elif name == 'efficientnet_b0':
        return build_efficientnet_b0_backbone(pretrained, out_dim)
    
    elif name == 'mobilevit_xs':
        return build_mobilevit_xs_backbone(pretrained, out_dim)

    else:
        raise ValueError(f"Unsupported backbone: {name}")


# =========================================================
# 9. 测试
# =========================================================
if __name__ == "__main__":
    x = torch.randn(1, 2, 129, 63)

    model_names = [
        'mobilenet_v2',
        'mobilenet_v3_small',
        'shufflenet_v2_x1_5',
        'efficientnet_b0',
    ]

    if HAS_TIMM:
        model_names.append('convnextv2_atto')

    for name in model_names:
        model = build_backbone(name, pretrained=False, out_dim=256)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters of model: {total_params}")
        y = model(x)
        print(f"{name}: input {x.shape} -> output {y.shape}")
        from thop import profile
        flops, params = profile(model, inputs=(x, ))
        print(f"FLOPs: {flops / 1e9:.2f} G") # 以 GFLOPs 为单位输出
