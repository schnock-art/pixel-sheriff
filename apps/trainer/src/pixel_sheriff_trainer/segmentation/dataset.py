from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class SegmentationSample:
    path: Path
    asset_id: str
    image_id: int
    width: int
    height: int


class SegmentationDataset(Dataset):
    """Parses COCO instances JSON → (image_tensor, mask_tensor).

    Rasterizes polygon annotations to pixel masks (H×W LongTensor, class indices).
    Supports polygon (segmentation field) and bbox fallback.
    Background class = 0. Category classes start at 1.
    """

    def __init__(
        self,
        samples: list[SegmentationSample],
        annotations: dict[int, list[dict[str, Any]]],
        cat_id_to_idx: dict[int, int],
        image_transform: Any,
        *,
        target_width: int,
        target_height: int,
    ) -> None:
        self.samples = samples
        self.annotations = annotations
        self.cat_id_to_idx = cat_id_to_idx
        self.image_transform = image_transform
        self.target_width = target_width
        self.target_height = target_height

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")

        orig_width, orig_height = image.size
        image = image.resize((self.target_width, self.target_height), Image.BILINEAR)
        tensor = self.image_transform(image)

        mask = Image.new("L", (self.target_width, self.target_height), 0)
        draw = ImageDraw.Draw(mask)
        scale_x = self.target_width / max(orig_width, 1)
        scale_y = self.target_height / max(orig_height, 1)

        anns = self.annotations.get(sample.image_id, [])
        for ann in anns:
            cat_id = int(ann.get("category_id", -1))
            class_idx = self.cat_id_to_idx.get(cat_id)
            if class_idx is None:
                continue
            fill_val = class_idx + 1  # 1-indexed (0 = background)

            segmentation = ann.get("segmentation")
            if isinstance(segmentation, list) and segmentation:
                for polygon in segmentation:
                    if not isinstance(polygon, list) or len(polygon) < 6:
                        continue
                    scaled_pts = [
                        (float(polygon[i]) * scale_x, float(polygon[i + 1]) * scale_y)
                        for i in range(0, len(polygon) - 1, 2)
                    ]
                    if len(scaled_pts) >= 3:
                        draw.polygon(scaled_pts, fill=fill_val)
            else:
                bbox = ann.get("bbox")
                if isinstance(bbox, list) and len(bbox) >= 4:
                    x, y, w, h = bbox
                    x0 = float(x) * scale_x
                    y0 = float(y) * scale_y
                    x1 = (float(x) + float(w)) * scale_x
                    y1 = (float(y) + float(h)) * scale_y
                    draw.rectangle([x0, y0, x1, y1], fill=fill_val)

        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        return tensor, mask_tensor


@dataclass
class LoadedSegmentationData:
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    num_classes: int
    class_names: list[str]
    class_order: list[str]
    train_count: int
    val_count: int
    skipped_unlabeled: int


def _extract_if_missing(zip_path: Path, workdir: Path) -> Path:
    dataset_dir = workdir / "dataset"
    if (dataset_dir / "coco_instances.json").exists():
        return dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as bundle:
        bundle.extractall(dataset_dir)
    return dataset_dir


def build_segmentation_loaders(
    *,
    export_zip_path: Path,
    workdir: Path,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> LoadedSegmentationData:
    dataset_dir = _extract_if_missing(export_zip_path, workdir)
    coco_path = dataset_dir / "coco_instances.json"
    if not coco_path.exists():
        raise ValueError("coco_instances.json is missing from export zip")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    categories = coco.get("categories", [])
    if not categories:
        raise ValueError("No categories in coco_instances.json")
    cat_id_to_idx: dict[int, int] = {}
    class_names: list[str] = []
    class_order: list[str] = []
    for idx, cat in enumerate(categories):
        cat_id = int(cat["id"])
        cat_id_to_idx[cat_id] = idx
        class_names.append(str(cat.get("name", f"class_{cat_id}")))
        class_order.append(str(cat_id))
    num_classes = len(categories)

    images_raw = coco.get("images", [])
    image_map: dict[int, dict[str, Any]] = {int(img["id"]): img for img in images_raw}

    annotations_raw = coco.get("annotations", [])
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations_raw:
        img_id = int(ann.get("image_id", -1))
        if img_id < 0:
            continue
        annotations_by_image.setdefault(img_id, []).append(ann)

    samples: list[SegmentationSample] = []
    skipped = 0
    for img_id, img_info in image_map.items():
        rel_path = str(img_info.get("file_name", ""))
        img_path = dataset_dir / rel_path
        if not img_path.exists():
            skipped += 1
            continue
        if img_id not in annotations_by_image:
            skipped += 1
            continue
        samples.append(SegmentationSample(
            path=img_path,
            asset_id=str(img_info.get("asset_id", img_id)),
            image_id=img_id,
            width=int(img_info.get("width", 224)),
            height=int(img_info.get("height", 224)),
        ))

    if not samples:
        raise ValueError("No annotated segmentation samples found in coco_instances.json")

    # Split
    manifest_path = dataset_dir / "manifest.json"
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = manifest.get("splits", {})
        asset_to_image: dict[str, int] = {
            str(img_info.get("asset_id", img_id)): img_id
            for img_id, img_info in image_map.items()
        }
        if isinstance(splits.get("train"), dict):
            for aid in splits["train"].get("asset_ids", []):
                img_id = asset_to_image.get(str(aid))
                if img_id is not None:
                    train_ids.add(img_id)
        if isinstance(splits.get("val"), dict):
            for aid in splits["val"].get("asset_ids", []):
                img_id = asset_to_image.get(str(aid))
                if img_id is not None:
                    val_ids.add(img_id)

    if train_ids or val_ids:
        train_samples = [s for s in samples if s.image_id in train_ids]
        val_samples = [s for s in samples if s.image_id in val_ids]
        if not val_samples:
            val_samples = train_samples[-max(1, len(train_samples) // 5):]
    else:
        import random
        shuffled = list(samples)
        random.Random(1337).shuffle(shuffled)
        split_at = max(1, int(len(shuffled) * 0.8))
        train_samples = shuffled[:split_at]
        val_samples = shuffled[split_at:] if len(shuffled) > 1 else shuffled

    # Input shape from model config
    input_cfg = model_config.get("input", {})
    raw_size = input_cfg.get("input_size", [224, 224]) if isinstance(input_cfg, dict) else [224, 224]
    if isinstance(raw_size, list) and len(raw_size) >= 2:
        target_width, target_height = int(raw_size[0]), int(raw_size[1])
    else:
        target_width, target_height = 224, 224

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = max(1, int(training_config.get("batch_size", 4)))

    train_dataset = SegmentationDataset(
        train_samples, annotations_by_image, cat_id_to_idx, image_transform,
        target_width=target_width, target_height=target_height,
    )
    val_dataset = SegmentationDataset(
        val_samples, annotations_by_image, cat_id_to_idx, image_transform,
        target_width=target_width, target_height=target_height,
    )

    train_loader: DataLoader[Any] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    return LoadedSegmentationData(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        class_names=class_names,
        class_order=class_order,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        skipped_unlabeled=skipped,
    )
