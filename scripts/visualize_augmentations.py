"""Draw augmented samples with their transformed boxes.

This is the visual counterpart to tests/test_geometry.py: the tests prove the
box/pixel agreement numerically, this proves it to the eye on real photographs.
Each row is one source image; the first column is the evaluation transform
(deterministic resize) and the rest are random training augmentations.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import CocoDetectionDataset, denormalize  # noqa: E402

SIZE = 384
COLUMNS = 4
ROWS = 4
PALETTE = {  # BGR, drawn with cv2
    "car": (58, 57, 230),
    "truck": (97, 162, 244),
    "traffic light": (143, 157, 42),
    "person": (157, 123, 69),
}
OUT = ROOT / "data" / "figures" / "augmented_samples.png"


def draw(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
         names: list[str]) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    for box, label in zip(boxes, labels):
        name = names[int(label)]
        colour = PALETTE.get(name, (200, 200, 200))
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(canvas, name[:3], (x1 + 1, max(y1 - 2, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)
    return canvas


def label_strip(text: str, width: int) -> np.ndarray:
    strip = np.full((18, width, 3), 32, dtype=np.uint8)
    cv2.putText(strip, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (235, 235, 235), 1, cv2.LINE_AA)
    return strip


def main() -> None:
    coco = ROOT / "data" / "annotations.json"
    images = ROOT / "data" / "images"

    train = CocoDetectionDataset(coco, images, size=SIZE, train=True)
    evaluation = CocoDetectionDataset(coco, images, size=SIZE, train=False)
    names = train.class_names
    print(f"{len(train)} images, classes {names}")

    # Prefer busy images: an augmentation bug is obvious with many boxes and
    # easy to miss with one.
    indices = sorted(range(len(train)),
                     key=lambda i: -len(train.records[i]["annotations"]))[:40]
    random.Random(7).shuffle(indices)
    indices = indices[:ROWS]

    rows = []
    for index in indices:
        tiles = []
        item = evaluation[index]
        tiles.append(np.vstack([
            label_strip(f"eval  {len(item['boxes'])} boxes", SIZE),
            draw(denormalize(item["image"]), item["boxes"].numpy(),
                 item["labels"].numpy(), names),
        ]))
        for _ in range(COLUMNS - 1):
            item = train[index]
            tiles.append(np.vstack([
                label_strip(f"augmented  {len(item['boxes'])} boxes", SIZE),
                draw(denormalize(item["image"]), item["boxes"].numpy(),
                     item["labels"].numpy(), names),
            ]))
        rows.append(np.hstack(tiles))

    grid = np.vstack(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), grid)
    print(f"wrote {OUT}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    random.seed(7)
    main()
