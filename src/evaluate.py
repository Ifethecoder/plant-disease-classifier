"""I evaluate on validation data and plot learning curves and predictions."""
import matplotlib

# I save figures to files so plotting works without a graphical display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import SequentialSampler

from .dataset import MEAN, STD


def classification_metrics(targets, predictions, num_classes):
    return {"accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, average="macro",
                                       labels=list(range(num_classes)), zero_division=0))}


def evaluate(model, loader, num_classes):
    # I keep prediction order aligned with images when displaying examples.
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Validation requires shuffle=False and drop_last=False")
    model.eval()
    device = next(model.parameters()).device
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            targets.extend(labels.tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(targets, predictions, num_classes)
    return metrics, targets, predictions


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="Training")
        axis.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.title())
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(dataset, targets, predictions, classes, path, mistakes_only=False):
    indices = [i for i in range(len(targets))
               if not mistakes_only or targets[i] != predictions[i]][:6]
    if not indices:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indices):
        image, _ = dataset[index]
        # I reverse normalization only for displaying the image.
        image = image.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(f"Actual: {classes[targets[index]]}\n"
                       f"Predicted: {classes[predictions[index]]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
