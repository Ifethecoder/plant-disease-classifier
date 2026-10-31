"""Single-image inference with the exact checkpoint preprocessing and labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .dataset import image_transform
from .evaluate import load_checkpoint


class Predictor:
    def __init__(self, model, classes: list[str], image_size=224):
        if not classes:
            raise ValueError("Class vocabulary cannot be empty")
        self.model = model.eval()
        self.classes = list(classes)
        self.transform = image_transform(image_size)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device="cpu"):
        model, state = load_checkpoint(path, device)
        return cls(model, state["classes"], state["config"]["image_size"])

    def predict(self, image: Image.Image | str | Path) -> dict:
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("Expected a PIL image or image file path")
        device = next(self.model.parameters()).device
        tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = self.model(tensor)
            if logits.shape != (1, len(self.classes)):
                raise ValueError("Model output does not match class vocabulary")
            probabilities = logits.softmax(1)
            confidence, index = probabilities.max(1)
        class_index = index.item()
        name = self.classes[class_index]
        crop, _, disease = name.partition("___")
        return {"class_index": class_index, "class_name": name, "crop": crop,
                "disease": disease, "confidence": confidence.item()}


def predict_image(image, model, classes: list[str], image_size=224) -> dict:
    """Return predicted crop/disease and softmax confidence for one leaf image."""
    return Predictor(model, classes, image_size).predict(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(Predictor.from_checkpoint(args.checkpoint, args.device).predict(args.image), indent=2))
