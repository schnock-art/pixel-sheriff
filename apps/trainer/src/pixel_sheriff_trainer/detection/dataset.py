from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class DetectionSample:
    path: Path
    asset_id: str
    image_id: int
    width: int
    height: int


class DetectionDataset(Dataset):
    """Parses COCO instances JSON → (image_tensor, target_dict).

    target_dict format matches torchvision detection models (RetinaNet):
      {"boxes": FloatTensor[N, 4], "labels": LongTensor[N]}
    Boxes are in [x_min, y_min, x_max, y_max] format (XYXY), pixel coordinates.
    """

    def __init__(
        self,
        samples: list[DetectionSample],
        annotations: dict[int, list[dict[str, Any]]],  # image_id → [ann, ...]
        transform: Any,
        *,
        target_width: int,
        target_height: int,
    ) -> None:
        self.samples = samples
        self.annotations = annotations
        self.transform = transform
        self.target_width = target_width
        self.target_height = target_height

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")

        orig_width, orig_height = image.size
        image = image.resize((self.target_width, self.target_height), Image.BILINEAR)
        tensor = self.transform(image)

        anns = self.annotations.get(sample.image_id, [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        scale_x = self.target_width / max(orig_width, 1)
        scale_y = self.target_height / max(orig_height, 1)

        for ann in anns:
            bbox = ann.get("bbox")
            cat = ann.get("category_id")
            if not isinstance(bbox, list) or len(bbox) < 4 or not isinstance(cat, int):
                continue
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            if w <= 0 or h <= 0:
                continue
            x_min = x * scale_x
            y_min = y * scale_y
            x_max = (x + w) * scale_x
            y_max = (y + h) * scale_y
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(cat))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        target: dict[str, Any] = {"boxes": boxes_tensor, "labels": labels_tensor}
        return tensor, target


@dataclass
class LoadedDetectionData:
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


def build_detection_loaders(
    *,
    export_zip_path: Path,
    workdir: Path,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> LoadedDetectionData:
    dataset_dir = _extract_if_missing(export_zip_path, workdir)
    coco_path = dataset_dir / "coco_instances.json"
    if not coco_path.exists():
        raise ValueError("coco_instances.json is missing from export zip")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    # Parse categories → class order + names
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

    # Parse images
    images_raw = coco.get("images", [])
    image_map: dict[int, dict[str, Any]] = {int(img["id"]): img for img in images_raw}

    # Parse annotations → per-image-id lists (remap cat_id to 1-indexed for RetinaNet)
    annotations_raw = coco.get("annotations", [])
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations_raw:
        img_id = int(ann.get("image_id", -1))
        cat_id = int(ann.get("category_id", -1))
        if img_id < 0 or cat_id not in cat_id_to_idx:
            continue
        remapped = dict(ann)
        remapped["category_id"] = cat_id_to_idx[cat_id] + 1  # 1-indexed (0 = background)
        annotations_by_image.setdefault(img_id, []).append(remapped)

    # Build samples list
    samples: list[DetectionSample] = []
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
        samples.append(DetectionSample(
            path=img_path,
            asset_id=str(img_info.get("asset_id", img_id)),
            image_id=img_id,
            width=int(img_info.get("width", 224)),
            height=int(img_info.get("height", 224)),
        ))

    if not samples:
        raise ValueError("No annotated detection samples found in coco_instances.json")

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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = max(1, int(training_config.get("batch_size", 4)))

    def collate_fn(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict[str, Any]]]:
        return [item[0] for item in batch], [item[1] for item in batch]

    train_dataset = DetectionDataset(
        train_samples, annotations_by_image, transform,
        target_width=target_width, target_height=target_height,
    )
    val_dataset = DetectionDataset(
        val_samples, annotations_by_image, transform,
        target_width=target_width, target_height=target_height,
    )

    train_loader: DataLoader[Any] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    return LoadedDetectionData(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        class_names=class_names,
        class_order=class_order,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        skipped_unlabeled=skipped,
    )
