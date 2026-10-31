import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import LeafDataset, make_splits, resolve_leaf_id


def fixture_metadata():
    train = [{"image_path": f"{c}/{g}/{j}.jpg", "label": c, "leaf_id": f"{c}-{g}"}
             for c in range(3) for g in range(20) for j in range(2)]
    return {"records": {"train": train, "test": [dict(train[0])]},
            "classes": ["a", "b", "c"], "revision": "synthetic"}


def test_grouped_split_is_reproducible_nested_and_test_disjoint():
    meta = fixture_metadata()
    small = make_splits(meta, 0.1)
    large = make_splits(meta, 0.3)
    assert small == make_splits(meta, 0.1)
    assert small["validation_indices"] == large["validation_indices"]
    assert set(small["train_indices"]) <= set(large["train_indices"])
    rows = meta["records"]["train"]
    groups = lambda indices: {rows[i]["leaf_id"] for i in indices}
    assert groups(small["train_indices"]).isdisjoint(groups(small["validation_indices"]))
    assert "0-0" not in groups(small["train_indices"] + small["validation_indices"])
    assert {rows[i]["label"] for i in small["train_indices"]} == {0, 1, 2}
    assert small["counts"]["quarantined"] == 2


def test_preprocessing_rgb_shape_and_normalization():
    data = LeafDataset([{"image": Image.fromarray(np.zeros((64, 80), dtype=np.uint8)), "label": 2}])
    x, y = data[0]
    assert x.shape == (3, 224, 224) and y == 2
    assert torch.isfinite(x).all()
    assert x[0, 0, 0].item() == pytest.approx(-0.485 / 0.229)


def test_leaf_map_and_invalid_fraction():
    path = "raw/color/Apple___healthy/uuid___Photo 001.JPG"
    assert resolve_leaf_id(path, {"photo 001": ["Apple___healthy:::1"]}) == "Apple___healthy:::1"
    assert resolve_leaf_id(path, {}) == "fallback_Photo 001"
    with pytest.raises(ValueError):
        make_splits(fixture_metadata(), 0)
