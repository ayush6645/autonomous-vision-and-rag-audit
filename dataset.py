"""COCO-backed PyTorch dataset with an explicit OpenCV geometry pipeline.

Design notes
------------
Every geometric augmentation is expressed as a 3x3 affine matrix. The matrices
are composed first and applied to the pixels exactly once with a single
`cv2.warpAffine` call, and the *same* composed matrix maps the boxes. This has
two benefits over chaining individual transforms:

* one interpolation instead of several, so repeated resampling cannot blur the
  image or accumulate rounding error;
* the image and its boxes cannot drift apart, because there is only one
  transform to get wrong.

Boxes are mapped by warping all four corners and taking the axis-aligned
bounds of the result. Warping only the top-left and bottom-right corners is a
common bug: it is correct for scale and translation but wrong for any rotation
or shear.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet statistics. The custom CNN is trained from scratch, but these are
# still a reasonable whitening of natural photographs and keep the input
# distribution comparable to published baselines.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MIN_VISIBLE_AREA = 0.30  # keep a box only if this fraction survives the warp
MIN_BOX_SIDE = 2.0  # pixels


# --------------------------------------------------------------------------
# Sample container
# --------------------------------------------------------------------------


@dataclass
class Sample:
    """One image with its boxes in absolute xyxy pixel coordinates."""

    image: np.ndarray  # HxWx3, uint8, RGB
    boxes: np.ndarray  # (N, 4) float32, xyxy
    labels: np.ndarray  # (N,) int64, zero-based class indices
    image_id: int

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the current image."""
        return self.image.shape[1], self.image.shape[0]


class Transform(Protocol):
    """Anything that maps a Sample to a Sample."""

    def __call__(self, sample: Sample, rng: random.Random) -> Sample: ...


# --------------------------------------------------------------------------
# Affine primitives
# --------------------------------------------------------------------------


def identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def translation(tx: float, ty: float) -> np.ndarray:
    matrix = identity()
    matrix[0, 2], matrix[1, 2] = tx, ty
    return matrix


def scaling(sx: float, sy: float) -> np.ndarray:
    matrix = identity()
    matrix[0, 0], matrix[1, 1] = sx, sy
    return matrix


def rotation(degrees: float, center: tuple[float, float]) -> np.ndarray:
    """Rotate about `center`, counter-clockwise in image coordinates."""
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    cx, cy = center
    matrix = identity()
    matrix[0, 0], matrix[0, 1] = cos, -sin
    matrix[1, 0], matrix[1, 1] = sin, cos
    # Translate the centre to the origin, rotate, translate back.
    matrix[0, 2] = cx - cos * cx + sin * cy
    matrix[1, 2] = cy - sin * cx - cos * cy
    return matrix


def shear(degrees_x: float, degrees_y: float,
          center: tuple[float, float]) -> np.ndarray:
    cx, cy = center
    matrix = identity()
    matrix[0, 1] = math.tan(math.radians(degrees_x))
    matrix[1, 0] = math.tan(math.radians(degrees_y))
    matrix[0, 2] = -matrix[0, 1] * cy
    matrix[1, 2] = -matrix[1, 0] * cx
    return matrix


def horizontal_flip(width: int) -> np.ndarray:
    """Mirror about the vertical centre line of a `width`-wide image."""
    matrix = identity()
    matrix[0, 0] = -1.0
    matrix[0, 2] = float(width)
    return matrix


def transform_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map xyxy boxes through a 3x3 affine matrix.

    All four corners are warped and the axis-aligned bounding box of the
    result is returned, which is the only correct treatment under rotation
    and shear.
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 4).astype(np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # (N, 4, 3) homogeneous corners: TL, TR, BR, BL.
    corners = np.stack([
        np.stack([x1, y1, np.ones_like(x1)], axis=1),
        np.stack([x2, y1, np.ones_like(x1)], axis=1),
        np.stack([x2, y2, np.ones_like(x1)], axis=1),
        np.stack([x1, y2, np.ones_like(x1)], axis=1),
    ], axis=1)

    warped = corners @ matrix.T  # (N, 4, 3)
    xs, ys = warped[:, :, 0], warped[:, :, 1]
    return np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)],
                    axis=1).astype(np.float32)


def clip_and_filter(boxes: np.ndarray, labels: np.ndarray, width: int,
                    height: int, original_areas: np.ndarray | None = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Clip boxes to the canvas and drop ones that barely survived.

    A box pushed mostly outside the frame by a warp is worse than no box: it
    teaches the model that a sliver of a car is a whole car.
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 4).astype(np.float32), labels

    clipped = boxes.copy()
    clipped[:, 0] = clipped[:, 0].clip(0, width)
    clipped[:, 1] = clipped[:, 1].clip(0, height)
    clipped[:, 2] = clipped[:, 2].clip(0, width)
    clipped[:, 3] = clipped[:, 3].clip(0, height)

    widths = clipped[:, 2] - clipped[:, 0]
    heights = clipped[:, 3] - clipped[:, 1]
    keep = (widths >= MIN_BOX_SIDE) & (heights >= MIN_BOX_SIDE)

    if original_areas is not None:
        areas = widths * heights
        safe = np.maximum(original_areas, 1e-6)
        keep &= (areas / safe) >= MIN_VISIBLE_AREA

    return clipped[keep].astype(np.float32), labels[keep]


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


class GeometricPipeline:
    """Compose resize, flip, rotation, scale, shear and shift into one warp."""

    def __init__(self, size: int, train: bool, degrees: float = 12.0,
                 scale: tuple[float, float] = (0.80, 1.20),
                 shear_degrees: float = 6.0, translate: float = 0.08,
                 flip_probability: float = 0.5) -> None:
        self.size = size
        self.train = train
        self.degrees = degrees
        self.scale = scale
        self.shear_degrees = shear_degrees
        self.translate = translate
        self.flip_probability = flip_probability

    def matrix_for(self, sample: Sample, rng: random.Random) -> np.ndarray:
        """Build the composed source -> destination matrix."""
        width, height = sample.size

        # 1. Letterbox-free resize onto a square canvas of `size`.
        ratio = min(self.size / width, self.size / height)
        matrix = scaling(ratio, ratio)
        # Centre the resized content on the square canvas.
        matrix = translation((self.size - width * ratio) / 2,
                             (self.size - height * ratio) / 2) @ matrix

        if not self.train:
            return matrix

        centre = (self.size / 2, self.size / 2)
        if rng.random() < self.flip_probability:
            matrix = horizontal_flip(self.size) @ matrix

        factor = rng.uniform(*self.scale)
        augment = translation(centre[0], centre[1]) @ scaling(factor, factor) \
            @ translation(-centre[0], -centre[1])
        augment = rotation(rng.uniform(-self.degrees, self.degrees),
                           centre) @ augment
        augment = shear(rng.uniform(-self.shear_degrees, self.shear_degrees),
                        rng.uniform(-self.shear_degrees, self.shear_degrees),
                        centre) @ augment
        shift = self.translate * self.size
        augment = translation(rng.uniform(-shift, shift),
                              rng.uniform(-shift, shift)) @ augment
        return augment @ matrix

    def __call__(self, sample: Sample, rng: random.Random) -> Sample:
        matrix = self.matrix_for(sample, rng)
        areas_before = ((sample.boxes[:, 2] - sample.boxes[:, 0])
                        * (sample.boxes[:, 3] - sample.boxes[:, 1])
                        ) if len(sample.boxes) else np.zeros(0, np.float32)

        image = cv2.warpAffine(
            sample.image, matrix[:2], (self.size, self.size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
        boxes = transform_boxes(sample.boxes, matrix)
        # Scale the reference areas by the determinant so the visibility test
        # compares like with like after a zoom.
        determinant = abs(np.linalg.det(matrix[:2, :2]))
        boxes, labels = clip_and_filter(
            boxes, sample.labels, self.size, self.size,
            areas_before * determinant if len(areas_before) else None,
        )
        return replace(sample, image=image, boxes=boxes, labels=labels)


class PhotometricJitter:
    """Brightness, contrast and saturation jitter. No geometry, so boxes
    are untouched by construction."""

    def __init__(self, brightness: float = 0.25, contrast: float = 0.25,
                 saturation: float = 0.25) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample: Sample, rng: random.Random) -> Sample:
        image = sample.image.astype(np.float32)
        image *= 1.0 + rng.uniform(-self.brightness, self.brightness)
        mean = image.mean()
        image = (image - mean) * (
            1.0 + rng.uniform(-self.contrast, self.contrast)) + mean

        grey = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        factor = 1.0 + rng.uniform(-self.saturation, self.saturation)
        image = grey[..., None] + (image - grey[..., None]) * factor

        return replace(sample, image=np.clip(image, 0, 255).astype(np.uint8))


def normalize(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> CHW float tensor, mean/std normalised."""
    array = image.astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class CocoDetectionDataset(Dataset):
    """Parses COCO JSON directly: no pycocotools at training time.

    The parser is deliberately explicit about the two COCO conventions that
    bite most often: `bbox` is [x, y, width, height] (not xyxy), and
    `category_id` is 1-based and need not be contiguous, so it is remapped to
    a dense zero-based index for the loss.
    """

    def __init__(self, coco_path: str | Path, image_dir: str | Path,
                 size: int = 384, train: bool = True,
                 image_ids: Sequence[int] | None = None,
                 seed: int = 0, min_boxes: int = 1) -> None:
        self.image_dir = Path(image_dir)
        self.size = size
        self.train = train
        self.seed = seed

        data = json.loads(Path(coco_path).read_text(encoding="utf-8"))

        categories = sorted(data["categories"], key=lambda c: c["id"])
        self.class_names = [c["name"] for c in categories]
        self.category_to_index = {c["id"]: i for i, c in enumerate(categories)}

        annotations: dict[int, list[dict]] = {}
        for annotation in data["annotations"]:
            if annotation.get("iscrowd", 0):
                continue
            annotations.setdefault(annotation["image_id"], []).append(annotation)

        wanted = set(image_ids) if image_ids is not None else None
        self.records: list[dict] = []
        for image in data["images"]:
            if wanted is not None and image["id"] not in wanted:
                continue
            boxes = annotations.get(image["id"], [])
            if len(boxes) < min_boxes:
                continue  # an image with no objects teaches nothing useful
            self.records.append({"image": image, "annotations": boxes})

        self.geometry = GeometricPipeline(size=size, train=train)
        self.jitter = PhotometricJitter() if train else None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def load_raw(self, index: int) -> Sample:
        """Read one image and convert its COCO boxes to xyxy."""
        record = self.records[index]
        info = record["image"]
        path = self.image_dir / info["file_name"]

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = [], []
        for annotation in record["annotations"]:
            x, y, w, h = annotation["bbox"]  # COCO stores xywh
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_to_index[annotation["category_id"]])

        return Sample(
            image=image,
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            labels=np.asarray(labels, dtype=np.int64),
            image_id=info["id"],
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.load_raw(index)

        # Deterministic per (epoch-independent) index when evaluating, and
        # reproducible across runs when training, because the seed is mixed
        # with the index rather than taken from global state.
        rng = random.Random((self.seed * 1_000_003) ^ index) if not self.train \
            else random.Random(random.getrandbits(32))

        sample = self.geometry(sample, rng)
        if self.jitter is not None:
            sample = self.jitter(sample, rng)

        return {
            "image": normalize(sample.image),
            "boxes": torch.from_numpy(sample.boxes),
            "labels": torch.from_numpy(sample.labels),
            "image_id": torch.tensor(sample.image_id, dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, object]:
    """Stack images; keep boxes as a list because counts vary per image."""
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_ids": torch.stack([item["image_id"] for item in batch]),
    }


def split_image_ids(coco_path: str | Path, fractions=(0.7, 0.15, 0.15),
                    seed: int = 0) -> tuple[list[int], list[int], list[int]]:
    """Deterministic train/val/test split over image ids."""
    data = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    ids = sorted(image["id"] for image in data["images"])
    random.Random(seed).shuffle(ids)

    n_train = int(len(ids) * fractions[0])
    n_val = int(len(ids) * fractions[1])
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Inverse of `normalize`, for visualisation and W&B logging."""
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * STD + MEAN
    return (np.clip(array, 0, 1) * 255).astype(np.uint8)
