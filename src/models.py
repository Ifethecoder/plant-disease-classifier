"""Identical ResNet-18 architecture, configurable initialization and freezing."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, pretrained: bool = False, freeze_backbone: bool = False):
    if num_classes < 2:
        raise ValueError("At least two classes are required")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # The training loop also holds frozen batch-normalization statistics fixed.
    model.freeze_backbone = freeze_backbone
    return model
