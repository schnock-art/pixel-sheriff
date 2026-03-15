from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


COLOR_JITTER_DEFAULTS = {
    "brightness": 0.15,
    "contrast": 0.15,
    "saturation": 0.1,
    "hue": 0.05,
}


@dataclass(frozen=True)
class AugmentationStep:
    type: str
    p: float
    params: dict[str, float]


def normalize_training_task(task: Any) -> str:
    normalized = str(task or "classification").strip().lower()
    if normalized in {"classification", "classification_single"}:
        return "classification"
    if normalized in {"detection", "bbox"}:
        return "detection"
    if normalized == "segmentation":
        return "segmentation"
    return "classification"


def preset_augmentation_steps(profile: Any) -> list[AugmentationStep]:
    normalized = str(profile or "none").strip().lower()
    if normalized == "light":
        return [AugmentationStep(type="horizontal_flip", p=0.5, params={})]
    if normalized == "medium":
        return [
            AugmentationStep(type="horizontal_flip", p=0.5, params={}),
            AugmentationStep(type="color_jitter", p=1.0, params=dict(COLOR_JITTER_DEFAULTS)),
        ]
    if normalized == "heavy":
        return [
            AugmentationStep(type="horizontal_flip", p=0.5, params={}),
            AugmentationStep(type="color_jitter", p=1.0, params=dict(COLOR_JITTER_DEFAULTS)),
            AugmentationStep(type="rotate", p=1.0, params={"degrees": 8.0}),
        ]
    return []


def _normalize_step(raw_step: Any) -> AugmentationStep | None:
    if not isinstance(raw_step, dict):
        return None
    step_type = str(raw_step.get("type") or "").strip().lower()
    if step_type not in {"horizontal_flip", "vertical_flip", "color_jitter", "rotate"}:
        return None
    raw_probability = raw_step.get("p", 1.0)
    try:
        probability = float(raw_probability)
    except (TypeError, ValueError):
        probability = 1.0
    params = raw_step.get("params")
    normalized_params = dict(params) if isinstance(params, dict) else {}
    safe_params: dict[str, float] = {}
    for key, value in normalized_params.items():
        try:
            safe_params[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return AugmentationStep(
        type=step_type,
        p=max(0.0, min(1.0, probability)),
        params=safe_params,
    )


def resolve_training_augmentation(training_config: dict[str, Any], task: Any) -> tuple[str, list[AugmentationStep]]:
    normalized_task = normalize_training_task(task or training_config.get("task"))
    raw_profile = training_config.get("augmentation_profile")
    profile = str(raw_profile or ("light" if normalized_task == "classification" else "none")).strip().lower()
    spec_version = training_config.get("augmentation_spec_version")

    if spec_version == 1:
        if profile == "custom":
            raw_steps = training_config.get("augmentation_steps")
            if not isinstance(raw_steps, list):
                return "custom", []
            return "custom", [step for step in (_normalize_step(raw) for raw in raw_steps) if step is not None]
        return profile, preset_augmentation_steps(profile)

    if normalized_task != "classification":
        return "none", []

    if profile not in {"none", "light", "medium", "heavy"}:
        profile = "none"
    return profile, preset_augmentation_steps(profile)


def apply_image_augmentation(image: Image.Image, steps: list[AugmentationStep]) -> Image.Image:
    augmented = image
    for step in steps:
        if random.random() > step.p:
            continue
        if step.type == "horizontal_flip":
            augmented = TF.hflip(augmented)
            continue
        if step.type == "vertical_flip":
            augmented = TF.vflip(augmented)
            continue
        if step.type == "color_jitter":
            jitter = transforms.ColorJitter(
                brightness=float(step.params.get("brightness", COLOR_JITTER_DEFAULTS["brightness"])),
                contrast=float(step.params.get("contrast", COLOR_JITTER_DEFAULTS["contrast"])),
                saturation=float(step.params.get("saturation", COLOR_JITTER_DEFAULTS["saturation"])),
                hue=float(step.params.get("hue", COLOR_JITTER_DEFAULTS["hue"])),
            )
            augmented = jitter(augmented)
            continue
        if step.type == "rotate":
            max_degrees = float(step.params.get("degrees", 0.0))
            if max_degrees <= 0:
                continue
            angle = random.uniform(-max_degrees, max_degrees)
            augmented = TF.rotate(augmented, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
    return augmented


def apply_detection_augmentation(
    image: Image.Image,
    boxes: list[list[float]],
    labels: list[int],
    steps: list[AugmentationStep],
) -> tuple[Image.Image, list[list[float]], list[int]]:
    augmented = image
    current_boxes = [list(box) for box in boxes]
    current_labels = list(labels)
    for step in steps:
        if random.random() > step.p:
            continue
        width, height = augmented.size
        if step.type == "horizontal_flip":
            augmented = TF.hflip(augmented)
            current_boxes = [[width - x_max, y_min, width - x_min, y_max] for x_min, y_min, x_max, y_max in current_boxes]
            continue
        if step.type == "vertical_flip":
            augmented = TF.vflip(augmented)
            current_boxes = [[x_min, height - y_max, x_max, height - y_min] for x_min, y_min, x_max, y_max in current_boxes]
            continue
        if step.type == "color_jitter":
            jitter = transforms.ColorJitter(
                brightness=float(step.params.get("brightness", COLOR_JITTER_DEFAULTS["brightness"])),
                contrast=float(step.params.get("contrast", COLOR_JITTER_DEFAULTS["contrast"])),
                saturation=float(step.params.get("saturation", COLOR_JITTER_DEFAULTS["saturation"])),
                hue=float(step.params.get("hue", COLOR_JITTER_DEFAULTS["hue"])),
            )
            augmented = jitter(augmented)
            continue
        if step.type == "rotate":
            max_degrees = float(step.params.get("degrees", 0.0))
            if max_degrees <= 0:
                continue
            angle = random.uniform(-max_degrees, max_degrees)
            augmented = TF.rotate(augmented, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            current_boxes, current_labels = _rotate_boxes(current_boxes, current_labels, width=width, height=height, angle=angle)
    return augmented, current_boxes, current_labels


def apply_segmentation_augmentation(
    image: Image.Image,
    mask: Image.Image,
    steps: list[AugmentationStep],
) -> tuple[Image.Image, Image.Image]:
    augmented_image = image
    augmented_mask = mask
    for step in steps:
        if random.random() > step.p:
            continue
        if step.type == "horizontal_flip":
            augmented_image = TF.hflip(augmented_image)
            augmented_mask = TF.hflip(augmented_mask)
            continue
        if step.type == "vertical_flip":
            augmented_image = TF.vflip(augmented_image)
            augmented_mask = TF.vflip(augmented_mask)
            continue
        if step.type == "color_jitter":
            jitter = transforms.ColorJitter(
                brightness=float(step.params.get("brightness", COLOR_JITTER_DEFAULTS["brightness"])),
                contrast=float(step.params.get("contrast", COLOR_JITTER_DEFAULTS["contrast"])),
                saturation=float(step.params.get("saturation", COLOR_JITTER_DEFAULTS["saturation"])),
                hue=float(step.params.get("hue", COLOR_JITTER_DEFAULTS["hue"])),
            )
            augmented_image = jitter(augmented_image)
            continue
        if step.type == "rotate":
            max_degrees = float(step.params.get("degrees", 0.0))
            if max_degrees <= 0:
                continue
            angle = random.uniform(-max_degrees, max_degrees)
            augmented_image = TF.rotate(augmented_image, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            augmented_mask = TF.rotate(augmented_mask, angle, interpolation=InterpolationMode.NEAREST, expand=False, fill=0)
    return augmented_image, augmented_mask


def _rotate_boxes(
    boxes: list[list[float]],
    labels: list[int],
    *,
    width: int,
    height: int,
    angle: float,
) -> tuple[list[list[float]], list[int]]:
    if not boxes:
        return [], []

    radians = math.radians(angle)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    center_x = width / 2.0
    center_y = height / 2.0

    rotated_boxes: list[list[float]] = []
    rotated_labels: list[int] = []
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
        rotated_corners = [
            _rotate_point(x, y, center_x=center_x, center_y=center_y, cos_theta=cos_theta, sin_theta=sin_theta)
            for x, y in corners
        ]
        xs = [min(max(point[0], 0.0), float(width)) for point in rotated_corners]
        ys = [min(max(point[1], 0.0), float(height)) for point in rotated_corners]
        clipped = [min(xs), min(ys), max(xs), max(ys)]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        rotated_boxes.append(clipped)
        rotated_labels.append(label)
    return rotated_boxes, rotated_labels


def _rotate_point(
    x: float,
    y: float,
    *,
    center_x: float,
    center_y: float,
    cos_theta: float,
    sin_theta: float,
) -> tuple[float, float]:
    translated_x = x - center_x
    translated_y = y - center_y
    rotated_x = (translated_x * cos_theta) - (translated_y * sin_theta)
    rotated_y = (translated_x * sin_theta) + (translated_y * cos_theta)
    return rotated_x + center_x, rotated_y + center_y
