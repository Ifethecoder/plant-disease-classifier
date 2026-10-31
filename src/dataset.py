"""I use published color splits and keep photographs of each leaf together."""
from __future__ import annotations

import hashlib
import json
import math
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Image, Value
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms

REPO_ID = "mohanty/PlantVillage"
REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_leaf_id(path: str, leaf_map: dict) -> str:
    """I follow the pinned upstream loader's identifier rules."""
    class_name = path.split("/")[2]
    identifier = Path(path).name.replace("_final_masked", "").split("___")[-1]
    identifier = identifier.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        identifier = identifier.replace(ext, "")
    identifier = identifier.strip()
    options = leaf_map.get(identifier.lower(), [])
    if len(options) == 1:
        return options[0]
    return next((x for x in options if class_name in x), f"fallback_{identifier}")


def read_metadata_files(root):
    root = Path(root)
    leaf_map = json.loads((root / "leaf-map.json").read_text())
    paths = {split: (root / f"color_{split}.txt").read_text().splitlines()
             for split in ("train", "test")}
    classes = sorted({path.split("/")[2] for path in paths["train"]})
    records = {}
    for split, filenames in paths.items():
        if len(filenames) != len(set(filenames)):
            raise ValueError(f"Duplicate paths in {split}")
        records[split] = []
        for path in filenames:
            parts = path.split("/")
            if len(parts) != 4 or parts[:2] != ["raw", "color"] or ".." in parts:
                raise ValueError(f"Unexpected color path: {path}")
            records[split].append({"image_path": path, "label": classes.index(parts[2]),
                                   "leaf_id": resolve_leaf_id(path, leaf_map)})
    return {"classes": classes, "records": records, "revision": REVISION,
            "upstream_schema": ["image_path", "label", "leaf_id"],
            "published_counts": {name: len(rows) for name, rows in records.items()}}


def load_metadata(cache_dir):
    # I read the manifests because the upstream dataset uses a legacy loading script.
    root = Path(cache_dir) / "metadata"
    root.mkdir(parents=True, exist_ok=True)
    files = {"leaf-map.json": "leaf_grouping/leaf-map.json",
             "color_train.txt": "splits/color_train.txt", "color_test.txt": "splits/color_test.txt"}
    for local, remote in files.items():
        source = hf_hub_download(REPO_ID, remote, repo_type="dataset", revision=REVISION,
                                 cache_dir=str(Path(cache_dir) / "hub"))
        (root / local).write_bytes(Path(source).read_bytes())
    return read_metadata_files(root)


def make_splits(metadata: dict, fraction: float = 0.1, seed: int = 42,
                val_fraction: float = 0.2) -> dict:
    """I split by leaf within each class, then sample training leaf groups."""
    if not 0 < fraction <= 1 or not 0 < val_fraction < 1:
        raise ValueError("Fractions must be in (0, 1], with validation strictly below 1")
    records = metadata["records"]["train"]
    test_groups = {r["leaf_id"] for r in metadata["records"]["test"]}
    test_paths = {r["image_path"] for r in metadata["records"]["test"]}
    quarantine = [i for i, r in enumerate(records)
                  if r["leaf_id"] in test_groups or r["image_path"] in test_paths]
    excluded = set(quarantine)
    grouped = defaultdict(lambda: defaultdict(list))
    leaf_labels = {}
    for i, row in enumerate(records):
        if i in excluded:
            continue
        leaf = row["leaf_id"]
        if leaf in leaf_labels and leaf_labels[leaf] != row["label"]:
            raise ValueError(f"Leaf has inconsistent class labels: {leaf}")
        leaf_labels[leaf] = row["label"]
        grouped[row["label"]][leaf].append(i)
    rng = np.random.default_rng(seed)
    train, validation, pool = [], [], []
    for label in range(len(metadata["classes"])):
        groups = sorted(grouped[label])
        if len(groups) < 2:
            raise ValueError(f"Class {label} needs at least two development leaf groups")
        groups = list(rng.permutation(groups))
        n_val = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
        val_groups, train_groups = groups[:n_val], groups[n_val:]
        validation.extend(i for g in val_groups for i in grouped[label][g])
        pool.extend(i for g in train_groups for i in grouped[label][g])
        n_train = max(1, math.ceil(len(train_groups) * fraction))
        train.extend(i for g in train_groups[:n_train] for i in grouped[label][g])
    payload = {"revision": metadata["revision"], "seed": seed, "fraction": fraction,
               "val_fraction": val_fraction, "train_indices": sorted(train),
               "validation_indices": sorted(validation), "pool_indices": sorted(pool),
               "quarantine_indices": quarantine, "classes": metadata["classes"]}
    payload["metadata_sha256"] = hashlib.sha256(json.dumps(metadata["records"], sort_keys=True).encode()).hexdigest()
    payload["split_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload["counts"] = {"train": len(train), "validation": len(validation), "pool": len(pool),
                         "quarantined": len(quarantine), "test_reserved": len(test_paths)}
    payload["actual_image_fraction"] = len(train) / len(pool)
    payload["class_counts"] = {name: {"train": sum(records[i]["label"] == k for i in train),
                                     "validation": sum(records[i]["label"] == k for i in validation)}
                               for k, name in enumerate(metadata["classes"])}
    return payload


def materialize_development(metadata: dict, splits: dict, cache_dir: str | Path) -> Dataset:
    """I extract only the selected development images from the cached archive."""
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(REPO_ID, "data.zip", repo_type="dataset", revision=REVISION,
                              cache_dir=str(cache_dir / "hub"))
    records = metadata["records"]["train"]
    selected = sorted(set(splits["train_indices"] + splits["validation_indices"]))
    rows = []
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
        # The archive may have an enclosing directory before raw/color/.
        lookup = {}
        for name in members:
            pos = name.find("raw/color/")
            if pos >= 0 and not name.endswith("/"):
                key = name[pos:]
                if key in lookup:
                    raise ValueError(f"Ambiguous archive member {key}")
                lookup[key] = name
        for index in selected:
            row = records[index]
            target = cache_dir / "images" / row["image_path"]
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(lookup[row["image_path"]]) as source:
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    temporary.write_bytes(source.read())
                    temporary.replace(target)
            rows.append({**row, "source_index": index, "image": str(target)})
    features = Features({"image_path": Value("string"), "label": ClassLabel(names=metadata["classes"]),
                         "leaf_id": Value("string"), "source_index": Value("int64"), "image": Image()})
    return Dataset.from_list(rows, features=features)


def image_transform(image_size: int = 224):
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    # I use ImageNet normalization for both models to keep preprocessing consistent.
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


class LeafDataset(TorchDataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform if transform is not None else image_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = row["image"]
        if isinstance(image, (str, Path)):
            with PILImage.open(image) as opened:
                image = opened.convert("RGB")
        else:
            image = image.convert("RGB")
        return self.transform(image), int(row["label"])


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(development: Dataset, splits: dict, batch_size=32, seed=42,
                 num_workers=0, image_size=224):
    if batch_size < 2:
        raise ValueError("Use batch_size >= 2 for ResNet training batch normalization")
    lookup = {int(v): i for i, v in enumerate(development["source_index"])}
    loaders = []
    for name in ("train", "validation"):
        indices = [lookup[i] for i in splits[f"{name}_indices"]]
        data = LeafDataset(development.select(indices), image_transform(image_size))
        # I avoid a one-image training batch because of batch normalization.
        drop_last = name == "train" and len(data) % batch_size == 1
        if len(data) < 2 and name == "train":
            raise ValueError("Need at least two training images")
        loaders.append(DataLoader(data, batch_size=batch_size, shuffle=name == "train",
                                  num_workers=num_workers, worker_init_fn=seed_worker,
                                  generator=torch.Generator().manual_seed(seed),
                                  drop_last=drop_last, pin_memory=torch.cuda.is_available()))
    return tuple(loaders)
