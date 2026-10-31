"""Validation metrics, learning curves and reviewable prediction evidence."""
from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .dataset import MEAN, STD
from .models import build_model
from .train import TrainConfig, resolve_device, save_json


def classification_metrics(targets, predictions, num_classes: int) -> dict:
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.ndim != 1 or targets.size == 0 or targets.shape != predictions.shape:
        raise ValueError("Expected nonempty, equally sized one-dimensional label arrays")
    if not np.isin(targets, np.arange(num_classes)).all() or not np.isin(predictions, np.arange(num_classes)).all():
        raise ValueError("Class index outside class vocabulary")
    return {"accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, labels=list(range(num_classes)),
                                        average="macro", zero_division=0)),
            "samples": int(len(targets)),
            "confusion_matrix": confusion_matrix(targets, predictions,
                                                  labels=list(range(num_classes))).tolist()}


def load_checkpoint(path: str | Path, device="cpu"):
    device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(len(checkpoint["classes"]), pretrained=False,
                        freeze_backbone=checkpoint["config"].get("freeze_backbone", False))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def plot_history(history: list[dict], output: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            axis.plot([h["epoch"] for h in history], [h[f"{split}_{metric}"] for h in history],
                      marker="o", label=split)
        axis.set(xlabel="Epoch", ylabel=metric.title(), title=f"Training and validation {metric}")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_examples(dataset, targets, predictions, confidences, classes, indices, output):
    indices = list(indices)[:8]
    if not indices:
        return False
    columns = min(4, len(indices))
    fig, axes = plt.subplots(int(np.ceil(len(indices) / columns)), columns,
                             figsize=(4 * columns, 4 * int(np.ceil(len(indices) / columns))), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, i in zip(axes.flat, indices):
        image, _ = dataset[i]
        image = image.cpu().permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(f"Actual: {classes[targets[i]]}\nPredicted: {classes[predictions[i]]}\n"
                       f"Softmax: {confidences[i]:.2%}", fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def evaluate_checkpoint(checkpoint_path, validation_loader, output_dir=None) -> dict:
    """Use the fixed, sequential validation loader; this API never loads test data."""
    from torch.utils.data import SequentialSampler
    if not isinstance(validation_loader.sampler, SequentialSampler) or validation_loader.drop_last:
        raise ValueError("Evaluation needs a sequential loader with drop_last=False")
    checkpoint_path = Path(checkpoint_path)
    output = Path(output_dir) if output_dir else checkpoint_path.parent
    output.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(checkpoint_path, "auto")
    device = next(model.parameters()).device
    targets, predictions, confidences = [], [], []
    with torch.inference_mode():
        for images, labels in validation_loader:
            probabilities = model(images.to(device)).softmax(1)
            confidence, predicted = probabilities.max(1)
            targets.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    metrics = classification_metrics(targets, predictions, len(checkpoint["classes"]))
    metrics.update({"split": "validation", "selected_epoch": checkpoint["epoch"],
                    "split_sha256": checkpoint["split_sha256"], "classes": checkpoint["classes"]})
    save_json(output / "validation_metrics.json", metrics)
    records = getattr(validation_loader.dataset, "records", None)
    rows = []
    for i, (actual, predicted, confidence) in enumerate(zip(targets, predictions, confidences)):
        row = {"validation_index": i, "actual": checkpoint["classes"][actual],
               "predicted": checkpoint["classes"][predicted], "confidence": confidence,
               "correct": actual == predicted}
        if records is not None:
            row["image_path"] = records[i]["image_path"]
            row["source_index"] = int(records[i]["source_index"])
        rows.append(row)
    with (output / "validation_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_examples(validation_loader.dataset, targets, predictions, confidences,
                  checkpoint["classes"], range(min(8, len(targets))), output / "examples.png")
    wrong = [i for i, (a, p) in enumerate(zip(targets, predictions)) if a != p]
    if not plot_examples(validation_loader.dataset, targets, predictions, confidences,
                         checkpoint["classes"], wrong, output / "misclassifications.png"):
        (output / "misclassifications.txt").write_text("No misclassifications in this validation run.\n")
    history_path = checkpoint_path.parent / "history.json"
    if history_path.exists():
        plot_history(json.loads(history_path.read_text()), output / "learning_curves.png")
    return metrics


def comparison_table(run_dirs: dict[str, Path], output_dir: str | Path) -> list[dict]:
    """Reject unequal budgets, split identities, class orders or preprocessing."""
    rows, reference = [], None
    for name, directory in run_dirs.items():
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        metrics = json.loads((directory / "validation_metrics.json").read_text())
        signature = {f.name: config[f.name] for f in fields(TrainConfig) if f.name != "pretrained"}
        signature.update({"split_sha256": metrics["split_sha256"], "classes": metrics["classes"]})
        if reference is not None and reference != signature:
            raise ValueError("Runs have different training budgets, splits or class mappings")
        reference = signature
        summary = json.loads((directory / "training_summary.json").read_text())
        rows.append({"model": name, "initialization": "ImageNet" if config["pretrained"] else "random",
                     "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                     "best_epoch": metrics["selected_epoch"], "epochs": config["epochs"],
                     "learning_rate": config["learning_rate"], "batch_size": config["batch_size"],
                     "validation_samples": metrics["samples"], "training_seconds": summary["seconds"]})
    if not rows:
        raise ValueError("At least one completed run is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Model | Validation accuracy | Macro F1 | Best epoch |",
             "|---|---:|---:|---:|"]
    lines.extend(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['best_epoch']} |" for r in rows)
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return rows
