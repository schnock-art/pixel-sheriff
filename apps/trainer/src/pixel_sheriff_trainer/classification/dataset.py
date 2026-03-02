from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ClassificationDataset(Dataset):
    def __init__(self, samples: list["ClassificationSample"], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, int(sample.label)


@dataclass(frozen=True)
class ClassificationSample:
    path: Path
    label: int
    asset_id: str
    relative_path: str


@dataclass
class LoadedClassificationData:
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    class_order: list[int]
    num_classes: int
    class_names: list[str]
    train_count: int
    val_count: int
    skipped_unlabeled: int
    val_samples: list[ClassificationSample]


def _normalization_from_model(model_config: dict[str, Any]) -> tuple[list[float] | None, list[float] | None]:
    input_config = model_config.get("input")
    if not isinstance(input_config, dict):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalization = input_config.get("normalization")
    if not isinstance(normalization, dict):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    norm_type = str(normalization.get("type") or "imagenet")
    if norm_type == "none":
        return None, None
    if norm_type == "custom":
        mean = normalization.get("mean")
        std = normalization.get("std")
        if isinstance(mean, list) and isinstance(std, list) and len(mean) == 3 and len(std) == 3:
            return [float(v) for v in mean], [float(v) for v in std]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _target_size_from_model(model_config: dict[str, Any]) -> tuple[int, int]:
    input_config = model_config.get("input")
    if not isinstance(input_config, dict):
        return (224, 224)
    raw_size = input_config.get("input_size")
    if isinstance(raw_size, list) and len(raw_size) == 2 and all(isinstance(v, int) and v > 0 for v in raw_size):
        width = int(raw_size[0])
        height = int(raw_size[1])
        return (width, height)
    return (224, 224)


def _extract_if_missing(zip_path: Path, workdir: Path) -> Path:
    dataset_dir = workdir / "dataset"
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        return dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as bundle:
        bundle.extractall(dataset_dir)
    return dataset_dir


def _asset_label_samples(
    manifest: dict[str, Any], dataset_dir: Path
) -> tuple[list[ClassificationSample], list[int], list[str], int]:
    label_schema = manifest.get("label_schema")
    if not isinstance(label_schema, dict):
        raise ValueError("manifest.label_schema is missing")
    class_order = label_schema.get("class_order")
    if not isinstance(class_order, list) or not class_order:
        raise ValueError("manifest.label_schema.class_order is missing")
    normalized_order: list[int] = []
    for raw in class_order:
        if isinstance(raw, int):
            normalized_order.append(raw)
    if not normalized_order:
        raise ValueError("manifest.label_schema.class_order is invalid")

    classes_raw = label_schema.get("classes")
    class_name_map: dict[int, str] = {}
    if isinstance(classes_raw, list):
        for row in classes_raw:
            if not isinstance(row, dict):
                continue
            class_id = row.get("id")
            if not isinstance(class_id, int):
                continue
            name = str(row.get("name") or f"class_{class_id}").strip()
            class_name_map[class_id] = name or f"class_{class_id}"
    class_names = [class_name_map.get(class_id, f"class_{class_id}") for class_id in normalized_order]
    class_index = {class_id: idx for idx, class_id in enumerate(normalized_order)}
    assets_raw = manifest.get("assets")
    annotations_raw = manifest.get("annotations")
    if not isinstance(assets_raw, list) or not isinstance(annotations_raw, list):
        raise ValueError("manifest assets/annotations are missing")

    asset_paths: dict[str, tuple[Path, str]] = {}
    for row in assets_raw:
        if not isinstance(row, dict):
            continue
        asset_id = row.get("asset_id")
        path = row.get("path")
        if isinstance(asset_id, str) and isinstance(path, str):
            asset_paths[asset_id] = (dataset_dir / path, path)

    skipped_unlabeled = 0
    samples: list[ClassificationSample] = []
    for row in annotations_raw:
        if not isinstance(row, dict):
            continue
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str):
            continue
        image_labels = (
            row.get("labels", {}).get("image")
            if isinstance(row.get("labels"), dict)
            else {}
        )
        primary_class_id = None
        if isinstance(image_labels, dict):
            raw_primary = image_labels.get("primary_class_id")
            if isinstance(raw_primary, int):
                primary_class_id = raw_primary
            elif isinstance(image_labels.get("class_ids"), list) and image_labels["class_ids"]:
                first = image_labels["class_ids"][0]
                if isinstance(first, int):
                    primary_class_id = first

        if not isinstance(primary_class_id, int):
            skipped_unlabeled += 1
            continue
        label_idx = class_index.get(primary_class_id)
        if label_idx is None:
            skipped_unlabeled += 1
            continue
        image_row = asset_paths.get(asset_id)
        if image_row is None:
            continue
        image_path, relative_path = image_row
        if not image_path.exists():
            continue
        samples.append(
            ClassificationSample(
                path=image_path,
                label=label_idx,
                asset_id=asset_id,
                relative_path=relative_path,
            )
        )
    return samples, normalized_order, class_names, skipped_unlabeled


def build_classification_loaders(
    *,
    export_zip_path: Path,
    workdir: Path,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> LoadedClassificationData:
    dataset_dir = _extract_if_missing(export_zip_path, workdir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("manifest.json is missing from export")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    samples, class_order, class_names, skipped_unlabeled = _asset_label_samples(manifest, dataset_dir)
    if not samples:
        raise ValueError("No labeled classification samples were found in manifest")

    split_train_ids = []
    split_val_ids = []
    splits = manifest.get("splits")
    if isinstance(splits, dict):
        split_train = splits.get("train")
        split_val = splits.get("val")
        if isinstance(split_train, dict) and isinstance(split_train.get("asset_ids"), list):
            split_train_ids = [str(v) for v in split_train["asset_ids"] if isinstance(v, str)]
        if isinstance(split_val, dict) and isinstance(split_val.get("asset_ids"), list):
            split_val_ids = [str(v) for v in split_val["asset_ids"] if isinstance(v, str)]

    sample_map = {sample.asset_id: sample for sample in samples}
    train_samples: list[ClassificationSample]
    val_samples: list[ClassificationSample]
    if split_train_ids or split_val_ids:
        train_samples = [sample_map[asset_id] for asset_id in split_train_ids if asset_id in sample_map]
        val_samples = [sample_map[asset_id] for asset_id in split_val_ids if asset_id in sample_map]
        if not val_samples:
            val_samples = train_samples[-max(1, len(train_samples) // 5) :]
    else:
        seed = 1337
        advanced = training_config.get("advanced")
        if isinstance(advanced, dict) and isinstance(advanced.get("seed"), int):
            seed = int(advanced["seed"])
        raw_samples = list(samples)
        random.Random(seed).shuffle(raw_samples)
        split_at = int(len(raw_samples) * 0.8)
        split_at = max(1, min(len(raw_samples) - 1, split_at)) if len(raw_samples) > 1 else 1
        train_samples = raw_samples[:split_at]
        val_samples = raw_samples[split_at:] if len(raw_samples) > 1 else raw_samples

    width, height = _target_size_from_model(model_config)
    mean, std = _normalization_from_model(model_config)

    train_steps: list[Any] = [transforms.Resize((height, width))]
    augmentation = str(training_config.get("augmentation_profile") or "none")
    if augmentation in {"light", "medium", "heavy"}:
        train_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if augmentation in {"medium", "heavy"}:
        train_steps.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05))
    if augmentation == "heavy":
        train_steps.append(transforms.RandomRotation(degrees=8))
    train_steps.append(transforms.ToTensor())
    if mean is not None and std is not None:
        train_steps.append(transforms.Normalize(mean=mean, std=std))
    train_transform = transforms.Compose(train_steps)

    val_steps: list[Any] = [transforms.Resize((height, width)), transforms.ToTensor()]
    if mean is not None and std is not None:
        val_steps.append(transforms.Normalize(mean=mean, std=std))
    val_transform = transforms.Compose(val_steps)

    batch_size = int(training_config.get("batch_size") or 16)
    advanced = training_config.get("advanced")
    num_workers = 0
    if isinstance(advanced, dict) and isinstance(advanced.get("num_workers"), int):
        num_workers = max(0, int(advanced["num_workers"]))

    train_dataset = ClassificationDataset(train_samples, train_transform)
    val_dataset = ClassificationDataset(val_samples, val_transform)
    train_loader: DataLoader[Any] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader: DataLoader[Any] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return LoadedClassificationData(
        train_loader=train_loader,
        val_loader=val_loader,
        class_order=class_order,
        num_classes=len(class_order),
        class_names=class_names,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        skipped_unlabeled=skipped_unlabeled,
        val_samples=val_samples,
    )
