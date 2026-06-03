import json
import math
import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(PROJECT_DIR / ".torch"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from vessel_guided_models import (
    build_resnet34_vessel_4ch_classifier,
    build_resnet34_vessel_attn_classifier,
)


BACKBONE_ALIASES = {
    "convnext_tiny": "convnext_tiny",
    "convnext-tiny": "convnext_tiny",
    "convnexttiny": "convnext_tiny",
    "convnext_t": "convnext_tiny",
    "convnext-t": "convnext_tiny",
    "convnextt": "convnext_tiny",
    "resnet34": "resnet34",
    "resnet_34": "resnet34",
    "resnet-34": "resnet34",
    "resnet34_pdc_c_v15": "resnet34_pdc_c_v15",
    "resnet34-pdc-c-v15": "resnet34_pdc_c_v15",
    "resnet34_c_v15": "resnet34_pdc_c_v15",
    "resnet34-c-v15": "resnet34_pdc_c_v15",
    "resnet34_pdc_r_v15": "resnet34_pdc_r_v15",
    "resnet34-pdc-r-v15": "resnet34_pdc_r_v15",
    "resnet34_r_v15": "resnet34_pdc_r_v15",
    "resnet34-r-v15": "resnet34_pdc_r_v15",
    "resnet34_pdc_a_v15": "resnet34_pdc_a_v15",
    "resnet34-pdc-a-v15": "resnet34_pdc_a_v15",
    "resnet34_a_v15": "resnet34_pdc_a_v15",
    "resnet34-a-v15": "resnet34_pdc_a_v15",
    "resnet34_pdc_cvvv_x4": "resnet34_pdc_cvvv_x4",
    "resnet34-pdc-cvvv-x4": "resnet34_pdc_cvvv_x4",
    "resnet34_cvvv_x4": "resnet34_pdc_cvvv_x4",
    "resnet34-cvvv-x4": "resnet34_pdc_cvvv_x4",
    "resnet34_pdc_stage_c": "resnet34_pdc_stage_c",
    "resnet34-pdc-stage-c": "resnet34_pdc_stage_c",
    "resnet34_stage_c": "resnet34_pdc_stage_c",
    "resnet34-stage-c": "resnet34_pdc_stage_c",
    "resnet34_pdc_c_all": "resnet34_pdc_c_all",
    "resnet34-pdc-c-all": "resnet34_pdc_c_all",
    "resnet34_c_all": "resnet34_pdc_c_all",
    "resnet34-c-all": "resnet34_pdc_c_all",
    "resnet34_gpdc_c_v15": "resnet34_gpdc_c_v15",
    "resnet34-gpdc-c-v15": "resnet34_gpdc_c_v15",
    "resnet34_gpdc_c": "resnet34_gpdc_c_v15",
    "resnet34-gpdc-c": "resnet34_gpdc_c_v15",
    "resnet34_ska_gpdc_c_v15": "resnet34_ska_gpdc_c_v15",
    "resnet34-ska-gpdc-c-v15": "resnet34_ska_gpdc_c_v15",
    "ska_gpdcnet": "resnet34_ska_gpdc_c_v15",
    "ska-gpdcnet": "resnet34_ska_gpdc_c_v15",
    "ska_gpdc_net": "resnet34_ska_gpdc_c_v15",
    "ska-gpdc-net": "resnet34_ska_gpdc_c_v15",
    "resnet34_gpdc_c_v15_struct_prior": "resnet34_gpdc_c_v15_struct_prior",
    "resnet34-gpdc-c-v15-struct-prior": "resnet34_gpdc_c_v15_struct_prior",
    "resnet34_gpdc_c_v15_prior": "resnet34_gpdc_c_v15_struct_prior",
    "resnet34-gpdc-c-v15-prior": "resnet34_gpdc_c_v15_struct_prior",
    "resnet34_vessel_4ch": "resnet34_vessel_4ch",
    "resnet34-vessel-4ch": "resnet34_vessel_4ch",
    "resnet34_vessel4ch": "resnet34_vessel_4ch",
    "resnet34-vessel4ch": "resnet34_vessel_4ch",
    "resnet34_vessel_attn": "resnet34_vessel_attn",
    "resnet34-vessel-attn": "resnet34_vessel_attn",
    "resnet34_vessel_attention": "resnet34_vessel_attn",
    "resnet34-vessel-attention": "resnet34_vessel_attn",
    "resnet50": "resnet50",
    "resnet_50": "resnet50",
    "resnet-50": "resnet50",
    "resnet101": "resnet101",
    "resnet_101": "resnet101",
    "resnet-101": "resnet101",
    "densenet121": "densenet121",
    "densenet_121": "densenet121",
    "densenet-121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnetb0": "efficientnet_b0",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "mobilenet-v3-large": "mobilenet_v3_large",
    "mobilenetv3_large": "mobilenet_v3_large",
    "mobilenetv3large": "mobilenet_v3_large",
    "swin_tiny": "swin_tiny",
    "swin-tiny": "swin_tiny",
    "swintiny": "swin_tiny",
    "swin_t": "swin_tiny",
    "swin-t": "swin_tiny",
    "swint": "swin_tiny",
}

BACKBONE_CHOICES = tuple(sorted(set(BACKBONE_ALIASES.values())))

BACKBONE_MODEL_LABELS = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "resnet34": "ResNet34",
    "resnet34_pdc_c_v15": "ResNet34-C-[V]x15",
    "resnet34_pdc_r_v15": "ResNet34-R-[V]x15",
    "resnet34_pdc_a_v15": "ResNet34-A-[V]x15",
    "resnet34_pdc_cvvv_x4": "ResNet34-[CVVV]x4",
    "resnet34_pdc_stage_c": "ResNet34-Stage-C",
    "resnet34_pdc_c_all": "ResNet34-[C]x16",
    "resnet34_gpdc_c_v15": "ResNet34-gPDC-C-[V]x15",
    "resnet34_ska_gpdc_c_v15": "SKA-gPDCNet",
    "resnet34_gpdc_c_v15_struct_prior": "ResNet34-gPDC-C-[V]x15-StructPrior",
    "resnet34_vessel_4ch": "ResNet34-Vessel4Ch",
    "resnet34_vessel_attn": "ResNet34-VesselAttn",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "swin_tiny": "Swin-Tiny",
}

RESNET34_PDC_ABLATION_BACKBONES = (
    "resnet34",
    "resnet34_pdc_c_v15",
    "resnet34_pdc_r_v15",
    "resnet34_pdc_a_v15",
    "resnet34_pdc_cvvv_x4",
    "resnet34_pdc_stage_c",
    "resnet34_pdc_c_all",
)

RESNET34_PDC_SEQUENCES = {
    "resnet34_pdc_c_v15": ("C",) + ("V",) * 15,
    "resnet34_pdc_r_v15": ("R",) + ("V",) * 15,
    "resnet34_pdc_a_v15": ("A",) + ("V",) * 15,
    "resnet34_pdc_cvvv_x4": ("C", "V", "V", "V") * 4,
    "resnet34_pdc_stage_c": (
        "C",
        "V",
        "V",
        "C",
        "V",
        "V",
        "V",
        "C",
        "V",
        "V",
        "V",
        "V",
        "V",
        "C",
        "V",
        "V",
    ),
    "resnet34_pdc_c_all": ("C",) * 16,
}


def resolve_torchvision_weights(default_weights, use_pretrained: bool = True):
    return default_weights if use_pretrained else None


def normalize_backbone_name(backbone_name: str):
    if backbone_name is None:
        raise ValueError("backbone_name must not be None")
    key = str(backbone_name).strip().lower()
    if key in BACKBONE_ALIASES:
        return BACKBONE_ALIASES[key]
    raise ValueError(f"Unsupported backbone: {backbone_name}. Choices: {', '.join(BACKBONE_CHOICES)}")


def get_env_float(name: str, default: float):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def get_ska_loss_weights():
    return {
        "lambda_proto": get_env_float("SKA_LAMBDA_PROTO", 0.1),
        "lambda_attr": get_env_float("SKA_LAMBDA_ATTR", 0.1),
    }


def get_requested_backbone(default="resnet50"):
    return normalize_backbone_name(os.environ.get("CLASSIFIER_BACKBONE", default))


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_metadata(split_dir: Path):
    summary_path = split_dir / "split_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def resolve_sample_path(split_dir: Path, stored_path: str, metadata: dict):
    sample_path = Path(stored_path)
    if sample_path.is_absolute():
        return str(sample_path)

    dataset_root_from_split_dir = metadata.get("dataset_root_from_split_dir")
    if dataset_root_from_split_dir:
        dataset_root = (split_dir / dataset_root_from_split_dir).resolve()
        return str((dataset_root / sample_path).resolve())

    return str((split_dir / sample_path).resolve())


def load_splits(split_dir: Path):
    split_path = split_dir / "splits.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    split_df = pd.read_csv(split_path)
    required_columns = {"filepath", "class_name", "class_index", "subset"}
    missing = required_columns - set(split_df.columns)
    if missing:
        raise ValueError(f"split file is missing columns: {sorted(missing)}")
    metadata = load_split_metadata(split_dir)
    split_df["resolved_filepath"] = split_df["filepath"].apply(
        lambda stored_path: resolve_sample_path(split_dir, stored_path, metadata)
    )
    class_table = (
        split_df[["class_name", "class_index"]]
        .drop_duplicates()
        .sort_values("class_index")
        .reset_index(drop=True)
    )
    class_names = class_table["class_name"].tolist()
    return split_df, class_names


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_transforms(image_size, training: bool):
    transform_steps = [transforms.Resize(image_size)]
    if training:
        transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_steps.extend(
        [
            transforms.ToTensor(),
        ]
    )
    return transforms.Compose(transform_steps)


class ClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size, training: bool, vessel_prior_root=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = tuple(image_size)
        self.training = bool(training)
        self.vessel_prior_root = Path(vessel_prior_root).resolve() if vessel_prior_root else None
        self.transform = build_transforms(image_size, training=training) if self.vessel_prior_root is None else None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row.get("resolved_filepath", row["filepath"])).convert("RGB")
        if self.vessel_prior_root is None:
            image = self.transform(image)
        else:
            image = self._load_vessel_guided_tensor(row, image)
        label = int(row["class_index"])
        return image, label


    def _load_vessel_guided_tensor(self, row, image):
        relative_path = Path(str(row["filepath"])).with_suffix(".png")
        vessel_path = self.vessel_prior_root / relative_path
        if not vessel_path.exists():
            raise FileNotFoundError(
                f"Missing vessel prior for {row['filepath']}: expected {vessel_path}"
            )

        vessel = Image.open(vessel_path).convert("L")
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        vessel = vessel.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        if self.training and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            vessel = vessel.transpose(Image.FLIP_LEFT_RIGHT)

        image_tensor = transforms.ToTensor()(image)
        vessel_tensor = transforms.ToTensor()(vessel)
        return torch.cat([image_tensor, vessel_tensor], dim=0)


def build_dataloader(dataframe, image_size, batch_size, training, seed, num_workers, vessel_prior_root=None):
    dataset = ClassificationDataset(
        dataframe,
        image_size=image_size,
        training=training,
        vessel_prior_root=vessel_prior_root,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    drop_last = training and len(dataset) > 1 and (len(dataset) % batch_size == 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if training else None,
    )


def build_mlp_head(in_features: int, num_classes: int):
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(inplace=True),
        nn.Linear(16, num_classes),
    )


def _softplus_inverse(value: float):
    return math.log(math.exp(value) - 1.0)


def _logit(value: float):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class BackboneClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_name: str):
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name

    def forward(self, x):
        return self.backbone(x)

    def compute_loss(self, outputs, labels, criterion=None):
        if hasattr(self.backbone, "compute_loss"):
            return self.backbone.compute_loss(outputs, labels, criterion=criterion)
        logits = extract_logits(outputs)
        if criterion is None:
            return F.cross_entropy(logits, labels)
        return criterion(logits, labels)


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


class PDCConv2d(nn.Conv2d):
    """Pixel-difference 3x3 convolution used inside selected ResNet34 BasicBlocks."""

    _ANGULAR_INDEX = [3, 0, 1, 6, 4, 2, 7, 8, 5]
    _RADIAL_OUTER_INDEX = [0, 2, 4, 10, 14, 20, 22, 24]
    _RADIAL_INNER_INDEX = [6, 7, 8, 11, 13, 16, 17, 18]

    def __init__(self, *args, pdc_type: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdc_type = pdc_type.upper()
        if self.pdc_type not in {"C", "A", "R"}:
            raise ValueError(f"Unsupported PDC type: {pdc_type}")
        if self.kernel_size != (3, 3):
            raise ValueError("PDCConv2d currently supports only 3x3 kernels")
        if self.padding_mode != "zeros":
            raise ValueError("PDCConv2d currently supports only zero padding")

    def forward(self, input):
        if self.pdc_type == "C":
            return self._central_difference_forward(input)
        if self.pdc_type == "A":
            return self._angular_difference_forward(input)
        return self._radial_difference_forward(input)

    def _central_difference_forward(self, input):
        vanilla = F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        kernel_diff = self.weight.sum(dim=(2, 3), keepdim=True)
        central = F.conv2d(
            input,
            kernel_diff,
            None,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )
        return vanilla - central

    def _angular_difference_forward(self, input):
        out_channels, in_channels_per_group, _, _ = self.weight.shape
        flat_weight = self.weight.reshape(out_channels, in_channels_per_group, -1)
        angular_weight = flat_weight - flat_weight[:, :, self._ANGULAR_INDEX]
        angular_weight = angular_weight.reshape_as(self.weight)
        return F.conv2d(
            input,
            angular_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def _radial_difference_forward(self, input):
        out_channels, in_channels_per_group, _, _ = self.weight.shape
        flat_weight = self.weight.reshape(out_channels, in_channels_per_group, -1)
        radial_weight = self.weight.new_zeros(out_channels, in_channels_per_group, 25)
        radial_weight[:, :, self._RADIAL_OUTER_INDEX] = flat_weight[:, :, 1:]
        radial_weight[:, :, self._RADIAL_INNER_INDEX] = -flat_weight[:, :, 1:]
        radial_weight = radial_weight.reshape(out_channels, in_channels_per_group, 5, 5)
        radial_padding = (self.padding[0] * 2, self.padding[1] * 2)
        return F.conv2d(
            input,
            radial_weight,
            self.bias,
            self.stride,
            radial_padding,
            self.dilation,
            self.groups,
        )


def make_pdc_conv_from_conv(conv: nn.Conv2d, pdc_type: str):
    pdc_conv = PDCConv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        pdc_type=pdc_type,
    )
    with torch.no_grad():
        pdc_conv.weight.copy_(conv.weight)
        if conv.bias is not None:
            pdc_conv.bias.copy_(conv.bias)
    return pdc_conv


class GatedCentralDifferenceConv2d(nn.Module):
    """Two-branch gPDC-C layer for selected ResNet34 BasicBlock convolutions."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        alpha_init=0.5,
    ):
        super().__init__()
        normalized_kernel = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if normalized_kernel != (3, 3):
            raise ValueError("GatedCentralDifferenceConv2d currently supports only 3x3 kernels")
        if padding_mode != "zeros":
            raise ValueError("GatedCentralDifferenceConv2d currently supports only zero padding")
        if groups != 1:
            raise ValueError("GatedCentralDifferenceConv2d currently supports only groups=1")

        self.gradient_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.intensity_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        kernel = torch.zeros(3, 3)
        kernel[1, 1] = 4.0
        kernel[0, 1] = -1.0
        kernel[1, 0] = -1.0
        kernel[1, 2] = -1.0
        kernel[2, 1] = -1.0
        self.register_buffer("central_difference_kernel", kernel.view(1, 1, 3, 3))
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(alpha_init), dtype=torch.float32))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, x):
        difference_kernel = self.central_difference_kernel.repeat(x.shape[1], 1, 1, 1)
        gradient_features = F.conv2d(x, difference_kernel, groups=x.shape[1], padding=1)
        gradient_out = self.gradient_conv(gradient_features)
        intensity_out = self.intensity_conv(x)
        alpha = self.alpha
        return alpha * gradient_out + (1.0 - alpha) * intensity_out


def make_gpdc_c_conv_from_conv(conv: nn.Conv2d):
    gpdc_conv = GatedCentralDifferenceConv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        gpdc_conv.gradient_conv.weight.copy_(conv.weight)
        gpdc_conv.intensity_conv.weight.copy_(conv.weight)
        if conv.bias is not None:
            gpdc_conv.gradient_conv.bias.copy_(conv.bias)
            gpdc_conv.intensity_conv.bias.copy_(conv.bias)
    return gpdc_conv


def iter_resnet34_basic_blocks(backbone: nn.Module):
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(backbone, layer_name)
        for block_index, block in enumerate(layer):
            yield layer_name, block_index, block


def apply_resnet34_pdc_sequence(backbone: nn.Module, sequence):
    if len(sequence) != 16:
        raise ValueError(f"ResNet34 PDC sequence must contain 16 block entries, got {len(sequence)}")

    for block_type, (_, _, block) in zip(sequence, iter_resnet34_basic_blocks(backbone)):
        block_type = block_type.upper()
        if block_type == "V":
            continue
        block.conv1 = make_pdc_conv_from_conv(block.conv1, block_type)
    return backbone


def apply_resnet34_gpdc_c_v15(backbone: nn.Module):
    for block_index, (_, _, block) in enumerate(iter_resnet34_basic_blocks(backbone)):
        if block_index == 0:
            block.conv1 = make_gpdc_c_conv_from_conv(block.conv1)
            break
    return backbone


def build_resnet34_classifier(num_classes: int, use_pretrained: bool = True):
    weights = resolve_torchvision_weights(models.ResNet34_Weights.DEFAULT, use_pretrained)
    backbone = models.resnet34(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = build_mlp_head(in_features, num_classes)
    return backbone


def build_resnet34_pdc_classifier(num_classes: int, backbone_name: str, use_pretrained: bool = True):
    backbone = build_resnet34_classifier(num_classes=num_classes, use_pretrained=use_pretrained)
    apply_resnet34_pdc_sequence(backbone, RESNET34_PDC_SEQUENCES[backbone_name])
    return backbone


def build_resnet34_gpdc_c_v15_classifier(num_classes: int, use_pretrained: bool = True):
    backbone = build_resnet34_classifier(num_classes=num_classes, use_pretrained=use_pretrained)
    apply_resnet34_gpdc_c_v15(backbone)
    return backbone


class ResNet34GPDCCV15StructPrior(nn.Module):
    """ResNet34-gPDC-C-[V]x15 classifier guided by a structure-prior attention map."""

    expected_input_channels = 4

    def __init__(self, num_classes: int, use_pretrained: bool = True, initial_gain: float = 0.5):
        super().__init__()
        backbone = build_resnet34_gpdc_c_v15_classifier(num_classes=num_classes, use_pretrained=use_pretrained)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

        initial = _softplus_inverse(initial_gain)
        self.stem_gain = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.layer1_gain = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.layer2_gain = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    @staticmethod
    def _apply_prior_attention(features, prior, raw_gain):
        attention = F.interpolate(prior, size=features.shape[-2:], mode="bilinear", align_corners=False)
        attention = attention.clamp(0.0, 1.0)
        gain = F.softplus(raw_gain)
        return features * (1.0 + gain * attention)

    def forward(self, x):
        if x.shape[1] != 4:
            raise ValueError(f"ResNet34GPDCCV15StructPrior expects 4 input channels, got {x.shape[1]}")

        rgb = x[:, :3]
        prior = x[:, 3:4]

        features = self.conv1(rgb)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self._apply_prior_attention(features, prior, self.stem_gain)

        features = self.layer1(features)
        features = self._apply_prior_attention(features, prior, self.layer1_gain)

        features = self.layer2(features)
        features = self._apply_prior_attention(features, prior, self.layer2_gain)

        features = self.layer3(features)
        features = self.layer4(features)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.fc(features)


def build_resnet34_gpdc_c_v15_struct_prior_classifier(num_classes: int, use_pretrained: bool = True):
    return ResNet34GPDCCV15StructPrior(num_classes=num_classes, use_pretrained=use_pretrained)


class ResNet34SKAGPDCCV15(nn.Module):
    """RGB-only SKA-gPDCNet classifier with explicit fundus semantic priors."""

    expected_input_channels = 3

    CLASS_NAMES = ("cataract", "diabetic_retinopathy", "glaucoma", "normal")
    ATTRIBUTE_NAMES = (
        "global_haze_low_contrast",
        "reduced_vessel_clarity",
        "hemorrhage_exudate_like_lesions",
        "optic_disc_abnormality",
        "cup_disc_ratio_enlargement",
        "vascular_morphology_abnormality",
        "structural_integrity",
    )
    ANATOMY_NAMES = ("global", "vessel", "optic_disc", "posterior_pole")

    DISEASE_ATTRIBUTE_PRIOR = torch.tensor(
        [
            [0.95, 0.65, 0.05, 0.10, 0.05, 0.10, 0.35],  # cataract
            [0.25, 0.75, 0.95, 0.10, 0.05, 0.80, 0.30],  # diabetic_retinopathy
            [0.10, 0.35, 0.10, 0.90, 0.95, 0.35, 0.35],  # glaucoma
            [0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.95],  # normal
        ],
        dtype=torch.float32,
    )
    ATTRIBUTE_ANATOMY_PRIOR = torch.tensor(
        [
            [0.90, 0.10, 0.05, 0.25],  # global haze / low contrast
            [0.15, 0.90, 0.05, 0.35],  # reduced vessel clarity
            [0.25, 0.45, 0.05, 0.85],  # hemorrhage / exudate-like lesions
            [0.15, 0.05, 0.95, 0.25],  # optic disc abnormality
            [0.10, 0.05, 0.95, 0.20],  # cup-disc ratio enlargement
            [0.15, 0.95, 0.15, 0.45],  # vascular morphology abnormality
            [0.80, 0.35, 0.35, 0.70],  # structural integrity
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        num_classes: int,
        use_pretrained: bool = True,
        semantic_dim: int = 256,
        beta: float = 0.5,
        proto_temperature: float = 0.1,
        lambda_proto: float = 0.1,
        lambda_attr: float = 0.1,
    ):
        super().__init__()
        if num_classes != len(self.CLASS_NAMES):
            raise ValueError(
                f"SKA-gPDCNet expects {len(self.CLASS_NAMES)} classes "
                f"({', '.join(self.CLASS_NAMES)}), got {num_classes}"
            )

        weights = resolve_torchvision_weights(models.ResNet34_Weights.DEFAULT, use_pretrained)
        backbone = models.resnet34(weights=weights)
        apply_resnet34_gpdc_c_v15(backbone)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.semantic_dim = int(semantic_dim)
        self.beta = float(beta)
        self.proto_temperature = float(proto_temperature)
        self.lambda_proto = float(lambda_proto)
        self.lambda_attr = float(lambda_attr)

        num_attributes = len(self.ATTRIBUTE_NAMES)
        num_anatomy = len(self.ANATOMY_NAMES)
        self.attribute_embeddings = nn.Parameter(torch.empty(num_attributes, self.semantic_dim))
        self.anatomy_embeddings = nn.Parameter(torch.empty(num_anatomy, self.semantic_dim))
        nn.init.normal_(self.attribute_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.anatomy_embeddings, mean=0.0, std=0.02)

        self.register_buffer("disease_attribute_matrix", self.DISEASE_ATTRIBUTE_PRIOR.clone())
        self.register_buffer("attribute_anatomy_matrix", self.ATTRIBUTE_ANATOMY_PRIOR.clone())

        self.deep_projection = nn.Conv2d(512, self.semantic_dim, kernel_size=1, bias=False)
        self.query_projection = nn.Linear(self.semantic_dim, self.semantic_dim)
        self.key_projection = nn.Linear(self.semantic_dim, self.semantic_dim)
        self.value_projection = nn.Linear(self.semantic_dim, self.semantic_dim)
        self.semantic_gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.shallow_projection = nn.Sequential(
            nn.Linear(64, self.semantic_dim),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(self.semantic_dim * 2, self.semantic_dim),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.semantic_dim, num_classes)
        self.attribute_head = nn.Linear(self.semantic_dim, num_attributes)

    def build_semantic_tokens(self):
        anatomy_context = self.attribute_anatomy_matrix @ self.anatomy_embeddings
        semantic_tokens = self.attribute_embeddings + self.beta * anatomy_context
        semantic_tokens = F.normalize(semantic_tokens, dim=-1)
        semantic_prototypes = self.disease_attribute_matrix @ semantic_tokens
        semantic_prototypes = F.normalize(semantic_prototypes, dim=-1)
        return semantic_tokens, semantic_prototypes

    def forward(self, x):
        features = self.conv1(x)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)

        shallow_features = self.layer1(features)
        features = self.layer2(shallow_features)
        features = self.layer3(features)
        deep_features = self.layer4(features)

        visual_tokens = self.deep_projection(deep_features).flatten(2).transpose(1, 2)
        semantic_tokens, semantic_prototypes = self.build_semantic_tokens()

        query = self.query_projection(visual_tokens)
        key = self.key_projection(semantic_tokens)
        value = self.value_projection(semantic_tokens)
        attention = torch.softmax((query @ key.transpose(0, 1)) / math.sqrt(self.semantic_dim), dim=-1)
        semantic_context = attention @ value
        visual_tokens = visual_tokens + self.semantic_gamma * semantic_context
        deep_vector = visual_tokens.mean(dim=1)

        shallow_vector = F.adaptive_avg_pool2d(shallow_features, output_size=1).flatten(1)
        shallow_vector = self.shallow_projection(shallow_vector)
        embedding = self.fusion_head(torch.cat([shallow_vector, deep_vector], dim=1))

        return {
            "logits": self.classifier(embedding),
            "attribute_logits": self.attribute_head(embedding),
            "embedding": embedding,
            "semantic_prototypes": semantic_prototypes,
        }

    def compute_loss(self, outputs, labels, criterion=None):
        logits = extract_logits(outputs)
        classification_loss = F.cross_entropy(logits, labels) if criterion is None else criterion(logits, labels)

        embedding = F.normalize(outputs["embedding"], dim=-1)
        prototypes = F.normalize(outputs["semantic_prototypes"], dim=-1)
        proto_logits = (embedding @ prototypes.transpose(0, 1)) / self.proto_temperature
        prototype_loss = F.cross_entropy(proto_logits, labels)

        attribute_targets = self.disease_attribute_matrix[labels].to(dtype=outputs["attribute_logits"].dtype)
        attribute_loss = F.binary_cross_entropy_with_logits(outputs["attribute_logits"], attribute_targets)

        return classification_loss + self.lambda_proto * prototype_loss + self.lambda_attr * attribute_loss


def build_resnet34_ska_gpdc_c_v15_classifier(num_classes: int, use_pretrained: bool = True):
    return ResNet34SKAGPDCCV15(
        num_classes=num_classes,
        use_pretrained=use_pretrained,
        **get_ska_loss_weights(),
    )


def build_resnet34_pdc_c_v15_classifier(num_classes: int, use_pretrained: bool = True):
    return build_resnet34_pdc_classifier(
        num_classes=num_classes,
        backbone_name="resnet34_pdc_c_v15",
        use_pretrained=use_pretrained,
    )


def build_resnet34_pdc_r_v15_classifier(num_classes: int, use_pretrained: bool = True):
    return build_resnet34_pdc_classifier(
        num_classes=num_classes,
        backbone_name="resnet34_pdc_r_v15",
        use_pretrained=use_pretrained,
    )


def build_resnet34_pdc_a_v15_classifier(num_classes: int, use_pretrained: bool = True):
    return build_resnet34_pdc_classifier(
        num_classes=num_classes,
        backbone_name="resnet34_pdc_a_v15",
        use_pretrained=use_pretrained,
    )


def build_resnet34_pdc_cvvv_x4_classifier(num_classes: int, use_pretrained: bool = True):
    return build_resnet34_pdc_classifier(
        num_classes=num_classes,
        backbone_name="resnet34_pdc_cvvv_x4",
        use_pretrained=use_pretrained,
    )


def build_resnet34_pdc_stage_c_classifier(num_classes: int, use_pretrained: bool = True):
    return build_resnet34_pdc_classifier(
        num_classes=num_classes,
        backbone_name="resnet34_pdc_stage_c",
        use_pretrained=use_pretrained,
    )


def build_resnet34_pdc_c_all_classifier(num_classes: int, use_pretrained: bool = True):
    return build_resnet34_pdc_classifier(
        num_classes=num_classes,
        backbone_name="resnet34_pdc_c_all",
        use_pretrained=use_pretrained,
    )


def build_resnet50_classifier(num_classes: int, use_pretrained: bool = True):
    weights = resolve_torchvision_weights(models.ResNet50_Weights.DEFAULT, use_pretrained)
    backbone = models.resnet50(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = build_mlp_head(in_features, num_classes)
    return backbone


def build_resnet101_classifier(num_classes: int, use_pretrained: bool = True):
    weights = resolve_torchvision_weights(models.ResNet101_Weights.DEFAULT, use_pretrained)
    backbone = models.resnet101(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = build_mlp_head(in_features, num_classes)
    return backbone


def build_convnext_tiny_classifier(num_classes: int, use_pretrained: bool = True):
    weights = resolve_torchvision_weights(models.ConvNeXt_Tiny_Weights.DEFAULT, use_pretrained)
    backbone = models.convnext_tiny(weights=weights)
    classifier_layers = list(backbone.classifier.children())
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Sequential(*classifier_layers[:-1], build_mlp_head(in_features, num_classes))
    return backbone


def build_densenet121_classifier(num_classes: int, use_pretrained: bool = True):
    weights = resolve_torchvision_weights(models.DenseNet121_Weights.DEFAULT, use_pretrained)
    backbone = models.densenet121(weights=weights)
    in_features = backbone.classifier.in_features
    backbone.classifier = build_mlp_head(in_features, num_classes)
    return backbone


def build_efficientnet_b0_classifier(num_classes: int, use_pretrained: bool = True):
    weights = resolve_torchvision_weights(models.EfficientNet_B0_Weights.DEFAULT, use_pretrained)
    backbone = models.efficientnet_b0(weights=weights)
    classifier_layers = list(backbone.classifier.children())
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Sequential(*classifier_layers[:-1], build_mlp_head(in_features, num_classes))
    return backbone


def build_mobilenet_v3_large_classifier(num_classes: int, use_pretrained: bool = True):
    weights = resolve_torchvision_weights(models.MobileNet_V3_Large_Weights.DEFAULT, use_pretrained)
    backbone = models.mobilenet_v3_large(weights=weights)
    classifier_layers = list(backbone.classifier.children())
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Sequential(*classifier_layers[:-1], build_mlp_head(in_features, num_classes))
    return backbone


def build_swin_tiny_classifier(num_classes: int, use_pretrained: bool = True):
    weights = resolve_torchvision_weights(models.Swin_T_Weights.DEFAULT, use_pretrained)
    backbone = models.swin_t(weights=weights)
    in_features = backbone.head.in_features
    backbone.head = build_mlp_head(in_features, num_classes)
    return backbone


MODEL_BUILDERS = {
    "convnext_tiny": build_convnext_tiny_classifier,
    "resnet34": build_resnet34_classifier,
    "resnet34_pdc_c_v15": build_resnet34_pdc_c_v15_classifier,
    "resnet34_pdc_r_v15": build_resnet34_pdc_r_v15_classifier,
    "resnet34_pdc_a_v15": build_resnet34_pdc_a_v15_classifier,
    "resnet34_pdc_cvvv_x4": build_resnet34_pdc_cvvv_x4_classifier,
    "resnet34_pdc_stage_c": build_resnet34_pdc_stage_c_classifier,
    "resnet34_pdc_c_all": build_resnet34_pdc_c_all_classifier,
    "resnet34_gpdc_c_v15": build_resnet34_gpdc_c_v15_classifier,
    "resnet34_ska_gpdc_c_v15": build_resnet34_ska_gpdc_c_v15_classifier,
    "resnet34_gpdc_c_v15_struct_prior": build_resnet34_gpdc_c_v15_struct_prior_classifier,
    "resnet34_vessel_4ch": build_resnet34_vessel_4ch_classifier,
    "resnet34_vessel_attn": build_resnet34_vessel_attn_classifier,
    "resnet50": build_resnet50_classifier,
    "resnet101": build_resnet101_classifier,
    "densenet121": build_densenet121_classifier,
    "efficientnet_b0": build_efficientnet_b0_classifier,
    "mobilenet_v3_large": build_mobilenet_v3_large_classifier,
    "swin_tiny": build_swin_tiny_classifier,
}


def infer_backbone_name(checkpoint: dict):
    backbone_name = checkpoint.get("backbone_name")
    if backbone_name:
        return normalize_backbone_name(backbone_name)

    architecture = str(checkpoint.get("architecture", "")).lower()
    for alias, canonical in BACKBONE_ALIASES.items():
        if alias in architecture:
            return canonical
    return get_requested_backbone()


def create_model(num_classes: int, backbone_name: str = None, use_pretrained: bool = True):
    canonical_name = normalize_backbone_name(backbone_name or get_requested_backbone())
    backbone = MODEL_BUILDERS[canonical_name](num_classes=num_classes, use_pretrained=use_pretrained)
    return BackboneClassifier(backbone=backbone, backbone_name=canonical_name)


def save_checkpoint(
    path: Path,
    model,
    class_names,
    image_size,
    epoch,
    metrics,
    backbone_name: str = None,
    vessel_prior_root: str = None,
):
    canonical_name = normalize_backbone_name(backbone_name or get_requested_backbone())
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": list(image_size),
        "epoch": int(epoch),
        "metrics": metrics,
        "framework": "pytorch",
        "architecture": f"torchvision_{canonical_name}_custom_head",
        "backbone_name": canonical_name,
        "vessel_prior_root": str(vessel_prior_root) if vessel_prior_root else None,
    }
    if canonical_name == "resnet34_ska_gpdc_c_v15":
        checkpoint["ska_loss_weights"] = get_ska_loss_weights()
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, map_location, use_pretrained: bool = False):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    class_names = checkpoint["class_names"]
    image_size = tuple(checkpoint["image_size"])
    backbone_name = infer_backbone_name(checkpoint)
    model = create_model(
        num_classes=len(class_names),
        backbone_name=backbone_name,
        use_pretrained=use_pretrained,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["backbone_name"] = backbone_name
    return model, class_names, image_size, checkpoint


def save_model_summary(output_dir: Path, model, device, backbone_name: str = None):
    canonical_name = normalize_backbone_name(backbone_name or get_requested_backbone())
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    payload = {
        "device": str(device),
        "architecture": model.__class__.__name__,
        "backbone_name": canonical_name,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }
    if canonical_name == "resnet34_ska_gpdc_c_v15":
        payload["ska_loss_weights"] = get_ska_loss_weights()
    (output_dir / "model_summary.txt").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
