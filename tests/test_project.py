"""Core pipeline checks using small synthetic datasets on CPU."""
import json
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.dataset import (LeafDataset, make_loaders, make_splits,
                         load_images, resolve_leaf_id, seed_everything)
from src.evaluate import classification_metrics, evaluate, plot_history, plot_predictions
from src.models import build_model, load_model, predict_image
from src.train import train_model


def example_metadata():
    classes = ["Apple___healthy", "Tomato___healthy"]
    rows = [{"image_path": f"raw/color/{classes[c]}/{leaf}_{photo}.jpg",
             "label": c, "leaf_id": f"{c}-{leaf}"}
            for c in range(2) for leaf in range(10) for photo in range(2)]
    return {"classes": classes, "revision": "synthetic",
            "records": {"train": rows, "test": [rows[0]]}}


def test_splits_keep_leaves_together_and_subsets_nested():
    metadata = example_metadata()
    rows = metadata["records"]["train"]
    previous = set()
    for fraction in (.01, .03, .10, .30):
        split = make_splits(metadata, fraction=fraction)
        assert split == make_splits(metadata, fraction=fraction)
        train = set(split["train_indices"])
        val = set(split["validation_indices"])
        assert previous <= train
        assert val == set(make_splits(metadata)["validation_indices"])
        train_leaves = {rows[i]["leaf_id"] for i in train}
        val_leaves = {rows[i]["leaf_id"] for i in val}
        assert train_leaves.isdisjoint(val_leaves)
        assert "0-0" not in train_leaves | val_leaves
        assert {rows[i]["label"] for i in train} == {0, 1}
        assert split["counts"]["quarantined"] == 2
        previous = train
    with pytest.raises(ValueError):
        make_splits(metadata, fraction=0)


def test_leaf_identifiers():
    path = "raw/color/Apple___healthy/uuid___Photo 001.JPG"
    assert resolve_leaf_id(path, {"photo 001": ["Apple___healthy:::1"]}) == "Apple___healthy:::1"
    assert resolve_leaf_id(path, {}) == "fallback_Photo 001"


def test_rgb_preprocessing():
    dataset = LeafDataset([{"image": Image.new("L", (40, 60)), "label": 1}])
    image, label = dataset[0]
    assert image.shape == (3, 224, 224) and label == 1
    assert image[0, 0, 0].item() == pytest.approx(-0.485 / 0.229)


def test_model_shape_and_weight_selection():
    torch.set_num_threads(1)
    model = build_model(38).eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 38)
    assert all(p.requires_grad for p in model.parameters())
    with patch("src.models.resnet18", return_value=resnet18(weights=None)) as factory:
        frozen = build_model(38, pretrained=True, freeze_backbone=True)
    factory.assert_called_once_with(weights=ResNet18_Weights.IMAGENET1K_V1)
    assert not frozen.conv1.weight.requires_grad and frozen.fc.weight.requires_grad


def test_accuracy_and_macro_f1():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert metrics["accuracy"] == .75
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + .8) / 2)
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 3)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + .8) / 3)


def test_archive_training_checkpoint_and_evaluation(tmp_path, monkeypatch):
    seed_everything(42)
    torch.set_num_threads(1)
    metadata = example_metadata()
    split = make_splits(metadata, fraction=.3)
    image_bytes = BytesIO()
    Image.new("RGB", (64, 64), "green").save(image_bytes, format="JPEG")
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for row in metadata["records"]["train"]:
            zipped.writestr("folder/" + row["image_path"], image_bytes.getvalue())
    monkeypatch.setattr("src.dataset.hf_hub_download", lambda *args, **kwargs: str(archive))
    data = load_images(metadata, split, tmp_path / "cache")
    assert data.features["label"].names == metadata["classes"]
    excluded = tmp_path / "cache/images" / metadata["records"]["test"][0]["image_path"]
    assert not excluded.exists()
    train_loader, val_loader = make_loaders(data, split, batch_size=4, image_size=64)
    second_train, _ = make_loaders(data, split, batch_size=4, image_size=64)
    assert list(train_loader.sampler) == list(second_train.sampler)
    model = build_model(2)
    before = model.fc.weight.detach().clone()
    output = tmp_path / "run"
    history = train_model(model, train_loader, val_loader, metadata["classes"], output,
                          epochs=2, image_size=64)
    assert not torch.equal(before, model.fc.weight)
    best, checkpoint = load_model(output / "best.pt")
    weights = best.fc.weight.detach().clone()
    metrics, targets, predictions = evaluate(best, val_loader, 2)
    assert torch.equal(weights, best.fc.weight)
    assert metrics["accuracy"] == max(row["val_accuracy"] for row in history)
    assert checkpoint["epoch"] in (1, 2)
    assert json.loads((output / "history.json").read_text()) == history
    assert len(targets) == len(val_loader.dataset)
    plot_history(history, output / "curves.png")
    assert plot_predictions(val_loader.dataset, targets, predictions, metadata["classes"], output / "examples.png")
    assert (output / "curves.png").read_bytes().startswith(b"\x89PNG")
    record = val_loader.dataset.records[0]
    result = predict_image(record["image"], best, checkpoint["classes"], checkpoint["image_size"])
    assert result["class_name"] == metadata["classes"][predictions[0]]


def test_inference_label_and_confidence(tmp_path):
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.copy_(torch.tensor([0., 2.]))
    image = Image.new("RGB", (64, 64))
    path = tmp_path / "leaf.png"
    image.save(path)
    classes = ["Apple___healthy", "Tomato___Early_blight"]
    result = predict_image(image, model, classes)
    assert result == predict_image(path, model, classes)
    assert result["class_name"] == "Tomato___Early_blight"
    assert result["crop"] == "Tomato" and result["disease"] == "Early_blight"
    assert result["confidence"] == pytest.approx(torch.tensor([0., 2.]).softmax(0)[1].item())


def test_notebook_is_valid_and_unexecuted():
    path = Path(__file__).resolve().parents[1] / "notebooks/progress_demo.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook_cell", "exec")
            assert cell.execution_count is None and cell.outputs == []
