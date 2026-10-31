"""CPU/CUDA training with matched budgets and best-validation checkpoints.

Run from the repository root: python -m src.train --help
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .dataset import (REVISION, load_metadata, make_loaders, make_splits,
                      materialize_development, seed_everything)
from .models import build_model


@dataclass(frozen=True)
class TrainConfig:
    pretrained: bool = False
    freeze_backbone: bool = False
    fraction: float = 0.1
    seed: int = 42
    val_fraction: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 5
    image_size: int = 224
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 2 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Need epochs >= 1, batch_size >= 2, learning_rate > 0")
        if not 0 < self.fraction <= 1 or not 0 < self.val_fraction < 1:
            raise ValueError("Invalid dataset fractions")
        if self.image_size < 32 or self.num_workers < 0 or self.seed < 0:
            raise ValueError("Invalid image size, workers, or seed")


def save_json(path: str | Path, data) -> None:
    """Atomic write prevents partial histories after interrupted runtimes."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: auto, cpu, cuda (or cuda:N)")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select a GPU runtime in Colab")
    return device


def run_epoch(model, loader, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    if training and getattr(model, "freeze_backbone", False):
        # Frozen backbone also means no running-statistic updates in batch norm.
        model.eval()
        model.fc.train()
    total_loss, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect inputs and learning rate")
            if training:
                loss.backward()
                optimizer.step()
            n = targets.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == targets).sum().item()
            count += n
    if not count:
        raise ValueError("Empty DataLoader")
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


def fit(model, train_loader, val_loader, config: TrainConfig, output_dir: str | Path,
        classes: list[str], split_sha256: str = "unspecified",
        split_manifest: dict | None = None) -> list[dict]:
    """Train and persist best state. Selection: accuracy, breaking ties by loss."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=config.learning_rate, weight_decay=0)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                          stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, git_dirty = None, None
    import importlib.metadata
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "datasets", "numpy", "scikit-learn", "Pillow")}
    save_json(output / "config.json", asdict(config))
    if split_manifest is not None:
        save_json(output / "splits.json", split_manifest)
    save_json(output / "environment.json", {"python": platform.python_version(),
              "platform": platform.platform(), "packages": packages, "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "git_commit": commit, "git_dirty": git_dirty, "dataset_revision": REVISION,
              "split_sha256": split_sha256, "optimizer": "Adam", "weight_decay": 0,
              "pretrained_weights": "IMAGENET1K_V1" if config.pretrained else None,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    history, best_key, best_epoch = [], (-1.0, float("-inf")), None
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
                        "train_samples": train_metrics["samples"],
                        "val_samples": val_metrics["samples"],
                        "seconds": time.perf_counter() - epoch_start})
        key = (val_metrics["accuracy"], -val_metrics["loss"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            checkpoint = {"model_state": model.state_dict(), "classes": classes,
                          "config": asdict(config), "epoch": epoch,
                          "val_accuracy": val_metrics["accuracy"], "val_loss": val_metrics["loss"],
                          "split_sha256": split_sha256}
            temp = output / "best.pt.tmp"
            torch.save(checkpoint, temp)
            temp.replace(output / "best.pt")
        save_json(output / "history.json", history)
        print(f"epoch {epoch}/{config.epochs}: train loss={train_metrics['loss']:.4f} "
              f"val loss={val_metrics['loss']:.4f} val accuracy={val_metrics['accuracy']:.4f}", flush=True)
    save_json(output / "training_summary.json", {"best_epoch": best_epoch,
              "best_val_accuracy": best_key[0], "seconds": time.perf_counter() - start})
    return history


def train_experiment(config: TrainConfig, output_dir: str | Path, cache_dir: str | Path):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    seed_everything(config.seed)
    metadata = load_metadata(Path(cache_dir) / "hub")
    splits = make_splits(metadata, config.fraction, config.seed, config.val_fraction)
    development = materialize_development(metadata, splits, cache_dir)
    loaders = make_loaders(development, splits, config.batch_size, config.seed,
                           config.num_workers, config.image_size)
    model = build_model(len(metadata["classes"]), config.pretrained, config.freeze_backbone)
    history = fit(model, *loaders, config, output_dir, metadata["classes"], splits["split_sha256"],
                  split_manifest=splits)
    return history


def parse_args(comparison=False):
    description = "Run a matched random/ImageNet ResNet-18 comparison." if comparison else __doc__
    parser = argparse.ArgumentParser(description=description)
    if not comparison:
        parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "plant-village")
    return parser.parse_args()


if __name__ == "__main__":
    args = vars(parse_args())
    output, cache = args.pop("output"), args.pop("cache_dir")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    train_experiment(TrainConfig(**args), output, cache)
