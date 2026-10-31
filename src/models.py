"""ResNet-18 with random or ImageNet initialization."""
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .dataset import image_transform


def build_model(num_classes, pretrained=False, freeze_backbone=False):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    # Replace ImageNet's 1,000 outputs with the PlantVillage classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.freeze_backbone = freeze_backbone
    return model


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def predict_image(image, model, classes, image_size=224):
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            image = source.convert("RGB")
    device = next(model.parameters()).device
    inputs = image_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = model(inputs).softmax(dim=1)
    confidence, index = probabilities[0].max(dim=0)
    name = classes[index.item()]
    crop, _, disease = name.partition("___")
    return {"class_name": name, "crop": crop, "disease": disease,
            "confidence": confidence.item()}
