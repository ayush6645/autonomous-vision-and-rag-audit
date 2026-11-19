"""Train a custom anchor-free detector on the scraped traffic dataset.

Architecture
------------
This is a single-shot, anchor-free detector in the CenterNet family, written
from scratch: no ResNet, no torchvision backbone, no pretrained weights. An
object is represented by one peak in a per-class heatmap plus two regressions
(box size and a sub-pixel centre offset).

Why this design for *this* dataset:

* The images are crowded -- a mean of 19 boxes and up to 114 per image. An
  anchor-based head would need anchor matching and NMS tuning before it did
  anything at all; a heatmap peak per object sidesteps both, and the loss is
  dense so every pixel contributes gradient from the first step.
* The objects are small. The median box covers 0.2% of its image, so at 384px
  input the median object is ~17px across. The neck therefore fuses the /16
  and /8 stages back up to stride 4, giving a 96x96 output grid where a 17px
  object still occupies four cells. A plain classification-style downsampling
  stack would have thrown those objects away.
* Batches are small (8-16 on the available hardware), which makes BatchNorm's
  running statistics noisy and the train/eval gap unpredictable. Every
  normalisation here is GroupNorm, which is batch-size independent.
* Width doubles as resolution halves, the standard compute-balanced choice,
  and two residual connections keep gradients flowing through the deepest
  part of the stack without the parameter cost of full residual blocks
  everywhere.

Everything below -- the loop, the loss, the schedule, the early stopping and
the evaluation -- is written out explicitly rather than delegated to a Trainer.

Usage
-----
    python train.py --epochs 60 --batch-size 8
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    CocoDetectionDataset,
    collate,
    denormalize,
    split_image_ids,
)

ROOT = Path(__file__).resolve().parent
COCO_PATH = ROOT / "data" / "annotations.json"
IMAGE_DIR = ROOT / "data" / "images"
RUN_DIR = ROOT / "runs"
CHECKPOINT_DIR = ROOT / "checkpoints"

STRIDE = 4  # output grid is INPUT // STRIDE


# ==========================================================================
# Model
# ==========================================================================


def conv_block(in_channels: int, out_channels: int, stride: int = 1,
               groups: int = 8) -> nn.Sequential:
    """3x3 conv -> GroupNorm -> SiLU.

    GroupNorm rather than BatchNorm because the batch size here is 8-16;
    BatchNorm's running statistics are unreliable at that size and create a
    train/eval mismatch that is easy to mistake for overfitting.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1,
                  bias=False),
        nn.GroupNorm(min(groups, out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    """Two 3x3 convs with an identity skip. Used only at the two deepest
    stages, where the gradient path is longest."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = conv_block(channels, channels)
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.second(self.first(x)))


class CentreNet(nn.Module):
    """Custom anchor-free detector.

    Encoder strides:  1 -> /2 -> /4 -> /8 -> /16
    Decoder:          /16 -> /8 -> /4, fusing the matching encoder stage.
    Heads at /4:      heatmap (num_classes), size (2), offset (2).
    """

    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__()
        self.num_classes = num_classes
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8

        # --- encoder -------------------------------------------------
        self.stem = nn.Sequential(          # /2
            conv_block(3, c1, stride=2),
            conv_block(c1, c1),
        )
        self.stage2 = nn.Sequential(        # /4
            conv_block(c1, c2, stride=2),
            conv_block(c2, c2),
        )
        self.stage3 = nn.Sequential(        # /8
            conv_block(c2, c3, stride=2),
            ResidualBlock(c3),
        )
        self.stage4 = nn.Sequential(        # /16
            conv_block(c3, c4, stride=2),
            ResidualBlock(c4),
        )

        # --- decoder -------------------------------------------------
        # Bilinear upsample + conv rather than ConvTranspose2d: transposed
        # convolutions produce checkerboard artefacts, which on a heatmap
        # show up as spurious peaks.
        self.up3 = conv_block(c4, c3)
        self.fuse3 = conv_block(c3, c3)
        self.up2 = conv_block(c3, c2)
        self.fuse2 = conv_block(c2, c2)

        # --- heads ---------------------------------------------------
        self.heatmap_head = self._head(c2, num_classes)
        self.size_head = self._head(c2, 2)
        self.offset_head = self._head(c2, 2)

        self._initialise()

    @staticmethod
    def _head(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            conv_block(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # The prediction layers are the exception: a fan_out Kaiming weight on
        # a 1x1 head makes the output swing far enough that the bias below is
        # irrelevant. Near-zero weights let each head start at its bias.
        for head in (self.heatmap_head, self.size_head, self.offset_head):
            nn.init.normal_(head[-1].weight, std=1e-3)
            nn.init.zeros_(head[-1].bias)

        # Start the heatmap at p~0.01. Without this the focal loss spends its
        # first epochs undoing a confident all-foreground prediction, which on
        # a sparse target looks exactly like a diverging run.
        nn.init.constant_(self.heatmap_head[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        d3 = F.interpolate(self.up3(s4), size=s3.shape[-2:], mode="bilinear",
                           align_corners=False)
        d3 = self.fuse3(d3 + s3)
        d2 = F.interpolate(self.up2(d3), size=s2.shape[-2:], mode="bilinear",
                           align_corners=False)
        d2 = self.fuse2(d2 + s2)

        return {
            "heatmap": self.heatmap_head(d2),  # logits
            "size": self.size_head(d2),
            "offset": self.offset_head(d2),
        }


# ==========================================================================
# Targets
# ==========================================================================


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7
                    ) -> float:
    """Radius at which a shifted box still overlaps the true box by
    `min_overlap`. Taken from the CornerNet derivation: three cases, each a
    quadratic, and the smallest root is the safe radius."""
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (-b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / (2 * a3)

    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    """Splat a 2D Gaussian into `heatmap` in place, taking the elementwise
    maximum so overlapping objects do not erase one another."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ys = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(ys[:, None] ** 2 + ys[None, :] ** 2)
                       / (2 * sigma ** 2))

    height, width = heatmap.shape
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return

    region = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    patch = kernel[radius - top:radius + bottom, radius - left:radius + right]
    torch.maximum(region, patch, out=region)


def build_targets(boxes_batch: list[torch.Tensor],
                  labels_batch: list[torch.Tensor], num_classes: int,
                  output_size: int, max_objects: int = 128
                  ) -> dict[str, torch.Tensor]:
    """Turn variable-length box lists into fixed-size training tensors.

    Size and offset are supervised only at the object centres, so they are
    gathered by index rather than stored densely -- a dense map would be
    mostly padding and would need its own mask anyway.
    """
    batch = len(boxes_batch)
    heatmap = torch.zeros(batch, num_classes, output_size, output_size)
    sizes = torch.zeros(batch, max_objects, 2)
    offsets = torch.zeros(batch, max_objects, 2)
    indices = torch.zeros(batch, max_objects, dtype=torch.long)
    mask = torch.zeros(batch, max_objects, dtype=torch.bool)

    for b, (boxes, labels) in enumerate(zip(boxes_batch, labels_batch)):
        for k, (box, label) in enumerate(zip(boxes, labels)):
            if k >= max_objects:
                break  # the 114-box image is the only one that hits this
            x1, y1, x2, y2 = (box / STRIDE).tolist()
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cxi, cyi = int(cx), int(cy)
            if not (0 <= cxi < output_size and 0 <= cyi < output_size):
                continue

            radius = max(1, int(gaussian_radius(h, w)))
            draw_gaussian(heatmap[b, int(label)], cxi, cyi, radius)

            # Log-space size regression. Raw width/height spans 4 to 96 grid
            # cells here, so L1 on raw values is dominated by a handful of
            # very large boxes and the small objects that make up most of
            # this dataset receive almost no gradient. In log space an error
            # is relative, so a 2-cell mistake on a 4-cell object matters as
            # much as a 48-cell mistake on a 96-cell one.
            sizes[b, k] = torch.tensor([math.log(w), math.log(h)])
            offsets[b, k] = torch.tensor([cx - cxi, cy - cyi])
            indices[b, k] = cyi * output_size + cxi
            mask[b, k] = True

    return {"heatmap": heatmap, "size": sizes, "offset": offsets,
            "index": indices, "mask": mask}


# ==========================================================================
# Loss
# ==========================================================================


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced focal loss (CornerNet).

    Plain BCE fails here: the target is ~0.1% positive, so predicting zero
    everywhere is already a near-perfect score. Two corrections are applied.
    `alpha` down-weights examples that are already correct. `beta` softens the
    penalty on negatives *near* a centre, using the Gaussian value itself --
    a pixel one cell from a car's centre is not really a negative, and
    punishing it as one makes the peaks unnecessarily sharp.
    """
    prediction = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positives = target.eq(1.0).float()
    negatives = 1.0 - positives

    positive_loss = (-torch.log(prediction)
                     * (1 - prediction) ** alpha * positives)
    negative_loss = (-torch.log(1 - prediction) * prediction ** alpha
                     * (1 - target) ** beta * negatives)

    count = positives.sum()
    total = positive_loss.sum() + negative_loss.sum()
    # With no objects in the batch only the negative term is defined.
    return total / count if count > 0 else negative_loss.sum()


def gather_at(feature: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Pick the (B, K, C) predictions at flat spatial indices (B, K)."""
    batch, channels = feature.shape[0], feature.shape[1]
    flat = feature.view(batch, channels, -1).permute(0, 2, 1)  # B, HW, C
    expanded = index.unsqueeze(2).expand(batch, index.shape[1], channels)
    return flat.gather(1, expanded)


@dataclass
class LossWeights:
    heatmap: float = 1.0
    size: float = 1.0   # sizes are regressed in log space, so the term is
                        # already order-1 and needs no extra down-weighting
    offset: float = 1.0


class DetectionLoss(nn.Module):
    """Focal loss on the heatmap plus masked L1 on size and offset."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(self, outputs: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]]:
        heat = focal_loss(outputs["heatmap"], targets["heatmap"])

        mask = targets["mask"]
        if mask.any():
            predicted_size = gather_at(outputs["size"], targets["index"])
            predicted_offset = gather_at(outputs["offset"], targets["index"])
            valid = mask.unsqueeze(2).float()
            denominator = valid.sum().clamp(min=1.0)
            size = (F.l1_loss(predicted_size * valid,
                              targets["size"] * valid, reduction="sum")
                    / denominator)
            offset = (F.l1_loss(predicted_offset * valid,
                                targets["offset"] * valid, reduction="sum")
                      / denominator)
        else:
            size = offset = torch.zeros((), device=heat.device)

        total = (self.weights.heatmap * heat + self.weights.size * size
                 + self.weights.offset * offset)
        return total, {"loss": total.item(), "heatmap": heat.item(),
                       "size": size.item(), "offset": offset.item()}


# ==========================================================================
# Decoding and evaluation
# ==========================================================================


@torch.no_grad()
def decode(outputs: dict[str, torch.Tensor], top_k: int = 100,
           threshold: float = 0.2
           ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Heatmap -> boxes in input pixels.

    The 3x3 max-pool is a peak test, and it replaces NMS entirely: a cell
    survives only if it is the local maximum, which is exactly the condition
    NMS approximates.
    """
    heatmap = torch.sigmoid(outputs["heatmap"])
    pooled = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    heatmap = heatmap * (pooled == heatmap).float()

    batch, classes, height, width = heatmap.shape
    scores, flat = heatmap.view(batch, -1).topk(min(top_k, classes * height * width))

    results = []
    for b in range(batch):
        keep = scores[b] > threshold
        picked = flat[b][keep]
        picked_scores = scores[b][keep]
        if len(picked) == 0:
            results.append((np.zeros((0, 4), np.float32),
                            np.zeros(0, np.int64), np.zeros(0, np.float32)))
            continue

        label = (picked // (height * width)).long()
        position = picked % (height * width)
        cy = (position // width).float()
        cx = (position % width).float()

        offset = gather_at(outputs["offset"][b:b + 1],
                           position.view(1, -1))[0]
        size = gather_at(outputs["size"][b:b + 1], position.view(1, -1))[0]
        cx = cx + offset[:, 0]
        cy = cy + offset[:, 1]
        # Sizes are regressed in log space (see build_targets); clamped
        # before exp so an early, badly-scaled prediction cannot overflow.
        w = size[:, 0].clamp(-5.0, 7.0).exp()
        h = size[:, 1].clamp(-5.0, 7.0).exp()

        boxes = torch.stack([cx - w / 2, cy - h / 2,
                             cx + w / 2, cy + h / 2], dim=1) * STRIDE
        results.append((boxes.cpu().numpy(), label.cpu().numpy(),
                        picked_scores.cpu().numpy()))
    return results


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - overlap
    return overlap / np.maximum(union, 1e-9)


def average_precision(records: list[tuple[float, bool]], positives: int
                      ) -> float:
    """11-point-free AP: sort by score, accumulate precision/recall, integrate
    with the all-point interpolation used by COCO/VOC2010."""
    if positives == 0 or not records:
        return 0.0
    records.sort(key=lambda r: -r[0])
    true_positives = np.cumsum([1.0 if hit else 0.0 for _, hit in records])
    false_positives = np.cumsum([0.0 if hit else 1.0 for _, hit in records])
    recall = true_positives / positives
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-9)

    # Make precision monotonically decreasing, then integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: DetectionLoss,
             device: torch.device, num_classes: int, output_size: int,
             iou_threshold: float = 0.5) -> dict[str, float]:
    """Validation loss plus AP@0.5 per class.

    Loss alone is a poor stopping signal for detection -- it is dominated by
    the heatmap term and can improve while the boxes get worse -- so mAP is
    computed too and used for model selection.
    """
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    per_class: dict[int, list[tuple[float, bool]]] = {c: [] for c in range(num_classes)}
    positives: dict[int, int] = {c: 0 for c in range(num_classes)}

    for batch in loader:
        images = batch["images"].to(device)
        targets = build_targets(batch["boxes"], batch["labels"], num_classes,
                                output_size)
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model(images)
        _, parts = criterion(outputs, targets)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

        for b, (boxes, labels, scores) in enumerate(decode(outputs)):
            truth_boxes = batch["boxes"][b].numpy()
            truth_labels = batch["labels"][b].numpy()
            for c in range(num_classes):
                gt = truth_boxes[truth_labels == c]
                positives[c] += len(gt)
                selected = labels == c
                if not selected.any():
                    continue
                predicted = boxes[selected]
                order = np.argsort(-scores[selected])
                predicted = predicted[order]
                predicted_scores = scores[selected][order]

                matched = np.zeros(len(gt), dtype=bool)
                ious = box_iou(predicted, gt)
                for i in range(len(predicted)):
                    hit = False
                    if len(gt):
                        best = int(np.argmax(ious[i]))
                        if ious[i, best] >= iou_threshold and not matched[best]:
                            matched[best] = True
                            hit = True
                    per_class[c].append((float(predicted_scores[i]), hit))

    metrics = {k: v / max(batches, 1) for k, v in totals.items()}
    aps = {c: average_precision(per_class[c], positives[c])
           for c in range(num_classes)}
    metrics["mAP50"] = float(np.mean(list(aps.values()))) if aps else 0.0
    for c, ap in aps.items():
        metrics[f"AP50/class_{c}"] = ap
    return metrics


# ==========================================================================
# Schedule and early stopping
# ==========================================================================


class WarmupCosine:
    """Linear warmup then cosine decay, stepped per optimiser update.

    Warmup matters here because the heatmap head starts at a deliberately
    biased initialisation; a full learning rate on step one undoes that bias
    before the backbone has learned anything.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 total_steps: int, base_lr: float, min_lr: float = 1e-6
                 ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(total_steps, self.warmup_steps + 1)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = ((self.step_count - self.warmup_steps)
                        / (self.total_steps - self.warmup_steps))
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


@dataclass
class EarlyStopping:
    """Stop when neither mAP nor validation loss has improved recently.

    mAP is the metric that matters, for the reason given in `evaluate`, but
    monitoring it alone is wrong early in a run. A first attempt did exactly
    that and stopped at epoch 13 because mAP had peaked at 0.005 -- pure
    noise -- at epoch 3, even though validation loss had fallen every single
    epoch since. The model was still plainly learning.

    So while mAP is below `map_floor` it is not yet a meaningful signal, and
    a falling validation loss also counts as progress. Once mAP rises above
    the floor it becomes the sole criterion, because at that point loss and
    detection quality can genuinely diverge.
    """

    patience: int = 10
    min_delta: float = 1e-4
    map_floor: float = 0.02
    best: float = -math.inf
    best_epoch: int = 0
    best_loss: float = math.inf
    bad_epochs: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, value: float, epoch: int,
               val_loss: float | None = None) -> bool:
        """Return True if this epoch is a new best worth checkpointing."""
        self.history.append(value)
        improved = value > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch = value, epoch

        loss_improved = (val_loss is not None
                         and val_loss < self.best_loss - self.min_delta)
        if loss_improved:
            self.best_loss = val_loss

        # Below the floor, mAP is noise: treat a falling loss as progress.
        progress = improved or (self.best < self.map_floor and loss_improved)
        self.bad_epochs = 0 if progress else self.bad_epochs + 1

        # Checkpoint on mAP once it means something, otherwise on loss.
        return improved or (self.best < self.map_floor and loss_improved)

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


# ==========================================================================
# Logging
# ==========================================================================


def draw_boxes(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
               names: list[str], colour: tuple[int, int, int]) -> np.ndarray:
    import cv2

    canvas = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(canvas, names[int(label)][:3], (x1 + 1, max(y1 - 2, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)
    return canvas


@torch.no_grad()
def log_samples(writer: SummaryWriter, model: nn.Module, batch: dict,
                names: list[str], device: torch.device, epoch: int,
                count: int = 4) -> None:
    """Log augmented inputs with ground truth (green) and predictions (red).

    The assignment asks for augmented samples each epoch, but ground truth
    alone would only show that augmentation runs. Overlaying predictions makes
    the same panel answer the question that actually matters during training:
    is the model learning these boxes?
    """
    model.eval()
    images = batch["images"][:count].to(device)
    predictions = decode(model(images))

    panels = []
    for i in range(len(images)):
        base = denormalize(images[i])
        base = draw_boxes(base, batch["boxes"][i].numpy(),
                          batch["labels"][i].numpy(), names, (40, 200, 40))
        boxes, labels, _ = predictions[i]
        base = draw_boxes(base, boxes, labels, names, (230, 60, 60))
        panels.append(torch.from_numpy(base.transpose(2, 0, 1)))

    if panels:
        writer.add_images("samples/green_truth_red_predicted",
                          torch.stack(panels), epoch)


# ==========================================================================
# Training
# ==========================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: DetectionLoss, optimizer: torch.optim.Optimizer,
                    scheduler: WarmupCosine, device: torch.device,
                    num_classes: int, output_size: int, epoch: int,
                    writer: SummaryWriter, clip: float = 5.0) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    started = time.time()

    for step, batch in enumerate(loader):
        images = batch["images"].to(device)
        targets = build_targets(batch["boxes"], batch["labels"], num_classes,
                                output_size)
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model(images)
        loss, parts = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # The focal term can spike when a batch happens to hold a crowd image;
        # clipping keeps one such batch from undoing an epoch.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        lr = scheduler.step()

        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

        if step % 10 == 0:
            global_step = epoch * len(loader) + step
            writer.add_scalar("batch/loss", parts["loss"], global_step)
            writer.add_scalar("batch/lr", lr, global_step)
            writer.add_scalar("batch/grad_norm", float(grad_norm), global_step)
            print(f"  epoch {epoch:>3} step {step:>4}/{len(loader)}  "
                  f"loss {parts['loss']:.4f}  hm {parts['heatmap']:.4f}  "
                  f"size {parts['size']:.3f}  lr {lr:.2e}", flush=True)

    metrics = {k: v / max(batches, 1) for k, v in totals.items()}
    metrics["seconds"] = time.time() - started
    return metrics


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, list[str]]:
    train_ids, val_ids, test_ids = split_image_ids(COCO_PATH, seed=args.seed)
    print(f"split: {len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test images")

    train_set = CocoDetectionDataset(COCO_PATH, IMAGE_DIR, size=args.size,
                                     train=True, image_ids=train_ids,
                                     seed=args.seed)
    val_set = CocoDetectionDataset(COCO_PATH, IMAGE_DIR, size=args.size,
                                   train=False, image_ids=val_ids,
                                   seed=args.seed)

    common = dict(collate_fn=collate, num_workers=args.workers,
                  persistent_workers=args.workers > 0,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, **common)
    return train_loader, val_loader, train_set.class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--width", type=int, default=32,
                        help="base channel count of the backbone")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="stop each epoch early; for smoke tests only")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    output_size = args.size // STRIDE

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_path = RUN_DIR / run_name
    writer = SummaryWriter(str(run_path))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_loaders(args)
    num_classes = len(class_names)
    # Captured now because --limit-batches later wraps the loader in an
    # object that has no .dataset to ask.
    train_image_count = len(train_loader.dataset)

    model = CentreNet(num_classes=num_classes, width=args.width).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {parameters / 1e6:.2f}M trainable parameters, "
          f"output grid {output_size}x{output_size}, classes {class_names}")

    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = args.limit_batches or len(train_loader)
    scheduler = WarmupCosine(
        optimizer,
        warmup_steps=int(args.warmup_epochs * steps_per_epoch),
        total_steps=args.epochs * steps_per_epoch,
        base_lr=args.lr,
    )
    stopper = EarlyStopping(patience=args.patience)

    if args.limit_batches:
        from itertools import islice

        class Limited:
            def __init__(self, loader, n):
                self.loader, self.n = loader, n

            def __iter__(self):
                return islice(iter(self.loader), self.n)

            def __len__(self):
                return min(self.n, len(self.loader))

        train_loader = Limited(train_loader, args.limit_batches)
        val_loader = Limited(val_loader, max(1, args.limit_batches // 2))

    # Record the run configuration next to the metrics. Without it a reader
    # of history.json -- including the Phase 3 auditor -- cannot tell a short
    # smoke test from a real run, and will blame the data for a low mAP that
    # is really just a truncated schedule.
    (run_path / "config.json").write_text(json.dumps({
        "run_name": run_name,
        "arguments": vars(args),
        "is_smoke_test": bool(args.limit_batches),
        "train_images": train_image_count,
        "batches_per_epoch": steps_per_epoch,
        "model_parameters": parameters,
        "class_names": class_names,
        "device": str(device),
    }, indent=2), encoding="utf-8")

    history = []
    sample_batch = next(iter(train_loader))

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion,
                                        optimizer, scheduler, device,
                                        num_classes, output_size, epoch,
                                        writer)
        val_metrics = evaluate(model, val_loader, criterion, device,
                               num_classes, output_size)

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        log_samples(writer, model, sample_batch, class_names, device, epoch)

        improved = stopper.update(val_metrics["mAP50"], epoch,
                                  val_metrics["loss"])
        print(f"epoch {epoch:>3}  train {train_metrics['loss']:.4f}  "
              f"val {val_metrics['loss']:.4f}  mAP50 {val_metrics['mAP50']:.4f}"
              f"{'  * best' if improved else ''}  "
              f"({train_metrics['seconds']:.0f}s)", flush=True)

        history.append({"epoch": epoch, "train": train_metrics,
                        "val": val_metrics})
        (run_path / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "mAP50": val_metrics["mAP50"],
                        "class_names": class_names,
                        "args": vars(args)},
                       CHECKPOINT_DIR / f"{run_name}_best.pt")

        if stopper.should_stop:
            print(f"early stop: no mAP improvement for {stopper.patience} "
                  f"epochs (best {stopper.best:.4f} at epoch "
                  f"{stopper.best_epoch})")
            break

    writer.close()
    print(f"\nbest mAP50 {stopper.best:.4f} at epoch {stopper.best_epoch}")
    print(f"checkpoint: {CHECKPOINT_DIR / f'{run_name}_best.pt'}")
    print(f"metrics log: {run_path / 'history.json'}")


if __name__ == "__main__":
    main()
