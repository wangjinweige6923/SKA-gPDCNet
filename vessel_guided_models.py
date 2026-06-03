import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


def _resolve_resnet34_weights(use_pretrained: bool):
    return models.ResNet34_Weights.DEFAULT if use_pretrained else None


def _build_mlp_head(in_features: int, num_classes: int):
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(inplace=True),
        nn.Linear(16, num_classes),
    )


def _softplus_inverse(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


class ResNet34Vessel4Ch(nn.Module):
    """ResNet34 classifier that consumes RGB plus one vessel-prior channel."""

    expected_input_channels = 4

    def __init__(self, num_classes: int, use_pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet34(weights=_resolve_resnet34_weights(use_pretrained))
        self._replace_first_conv()
        self.backbone.fc = _build_mlp_head(self.backbone.fc.in_features, num_classes)

    def _replace_first_conv(self):
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old_conv.weight)
            new_conv.weight[:, 3:4].copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.backbone.conv1 = new_conv

    def forward(self, x):
        if x.shape[1] != 4:
            raise ValueError(f"ResNet34Vessel4Ch expects 4 input channels, got {x.shape[1]}")
        return self.backbone(x)


class ResNet34VesselAttention(nn.Module):
    """ResNet34 classifier guided by a vessel-prior attention map."""

    expected_input_channels = 4

    def __init__(self, num_classes: int, use_pretrained: bool = True, initial_gain: float = 0.5):
        super().__init__()
        backbone = models.resnet34(weights=_resolve_resnet34_weights(use_pretrained))

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = _build_mlp_head(backbone.fc.in_features, num_classes)

        initial = _softplus_inverse(initial_gain)
        self.stem_gain = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.layer1_gain = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.layer2_gain = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    @staticmethod
    def _apply_vessel_attention(features, vessel, raw_gain):
        attention = F.interpolate(vessel, size=features.shape[-2:], mode="bilinear", align_corners=False)
        attention = attention.clamp(0.0, 1.0)
        gain = F.softplus(raw_gain)
        return features * (1.0 + gain * attention)

    def forward(self, x):
        if x.shape[1] != 4:
            raise ValueError(f"ResNet34VesselAttention expects 4 input channels, got {x.shape[1]}")

        rgb = x[:, :3]
        vessel = x[:, 3:4]

        features = self.conv1(rgb)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self._apply_vessel_attention(features, vessel, self.stem_gain)

        features = self.layer1(features)
        features = self._apply_vessel_attention(features, vessel, self.layer1_gain)

        features = self.layer2(features)
        features = self._apply_vessel_attention(features, vessel, self.layer2_gain)

        features = self.layer3(features)
        features = self.layer4(features)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.fc(features)


def build_resnet34_vessel_4ch_classifier(num_classes: int, use_pretrained: bool = True):
    return ResNet34Vessel4Ch(num_classes=num_classes, use_pretrained=use_pretrained)


def build_resnet34_vessel_attn_classifier(num_classes: int, use_pretrained: bool = True):
    return ResNet34VesselAttention(num_classes=num_classes, use_pretrained=use_pretrained)
