"""Training with cross-entropy and matching Adam settings for both models."""
import json
from pathlib import Path

import torch
from torch import nn


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training and getattr(model, "freeze_backbone", False):
        # A frozen backbone also needs fixed batch-normalization statistics.
        model.eval()
        model.fc.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


def train_model(model, train_loader, val_loader, classes, output_dir,
                epochs=5, learning_rate=0.001, device="cpu", image_size=224):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    settings = {"epochs": epochs, "learning_rate": learning_rate, "optimizer": "Adam",
                "batch_size": train_loader.batch_size, "device": str(device),
                "image_size": image_size, "freeze_backbone": getattr(model, "freeze_backbone", False)}
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    history = []
    best_score = (-1, float("-inf"))
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
                        "train_accuracy": train["accuracy"], "val_accuracy": val["accuracy"],
                        "train_samples": train["samples"], "val_samples": val["samples"]})
        # Select the best validation accuracy, using loss to break a tie.
        score = (val["accuracy"], -val["loss"])
        if score > best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "image_size": image_size, "epoch": epoch}, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch}/{epochs}: train loss {train['loss']:.4f}, "
              f"validation loss {val['loss']:.4f}, accuracy {val['accuracy']:.4f}")
    return history
