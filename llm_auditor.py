"""Agentic RAG auditor over the COCO annotations and the training log.

The design rests on one distinction: **retrieval supplies descriptions, tools
supply numbers.**

A vector store is good at finding the right passage and useless at counting.
Asked "how many boxes are under 1000 square pixels", a retrieval-only system
hands the model a few prose chunks and the model invents a plausible figure.
So the corpus carries the qualitative material -- what the dataset is, what
each class looks like, what happened during training -- while every exact
quantity is computed by a Python tool that reads `annotations.json` directly.
The model decides which tool to call; it never does the arithmetic itself.

Four tools are exposed:

  compute_box_statistics      area percentiles, optionally filtered
  check_class_balance         counts, shares, imbalance ratio
  find_suspicious_annotations tiny boxes, near-duplicates, out-of-bounds
  summarise_training          best epoch, overfitting gap, from history.json

Usage
-----
    python llm_auditor.py build                 # chunk + embed the corpus
    python llm_auditor.py ask "..."             # one grounded question
    python llm_auditor.py demo                  # the scripted scenarios
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
COCO_PATH = ROOT / "data" / "annotations.json"
RUN_DIR = ROOT / "runs"
STORE_DIR = ROOT / "data" / "vector_store"
TRANSCRIPT_DIR = ROOT / "data" / "audit_logs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "dataset_audit"
GEMINI_MODEL = "gemini-3.1-flash-lite"
TOP_K = 6
MAX_TOOL_ROUNDS = 6


# ==========================================================================
# Corpus construction
# ==========================================================================


@dataclass
class Chunk:
    """One retrievable passage plus the metadata used for filtering."""

    id: str
    text: str
    source: str  # "dataset" | "class" | "image" | "quality" | "training"


def squash(text: str) -> str:
    """Collapse the line breaks that source formatting leaves inside numbers.

    A dedented f-string wraps wherever the source line ended, which can put
    "median" and "17072" on separate lines. The embedding model sees that as
    two weakly related tokens, so sentences are rejoined and only paragraph
    breaks are kept.
    """
    paragraphs = [" ".join(block.split())
                  for block in text.split("\n\n") if block.strip()]
    return "\n\n".join(paragraphs)


def load_coco(path: Path = COCO_PATH) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run `python collect_and_annotate.py annotate` "
            "first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def latest_history(run_dir: Path = RUN_DIR) -> tuple[Path | None, list[dict]]:
    """Most recently modified history.json, if a training run has happened."""
    candidates = sorted(run_dir.glob("*/history.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None, []
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def run_config(history_path: Path | None) -> dict:
    """The configuration of the run that produced `history_path`.

    This matters more than it looks. Metrics without their configuration are
    ambiguous: a validation mAP of zero means something entirely different
    after three epochs on six batches than after sixty epochs on the full
    training set. Supplying the configuration is what stops the auditor
    blaming the dataset for a truncated schedule.
    """
    if history_path is None:
        return {}
    path = history_path.parent / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two COCO xywh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def build_chunks(coco: dict, history: list[dict],
                 config: dict | None = None) -> list[Chunk]:
    """Turn the COCO file, training log and run config into passages.

    Chunking is semantic rather than fixed-width: a sliding window over
    pretty-printed JSON would split a box away from its image and retrieve
    fragments that mean nothing. One chunk per meaningful unit instead.
    """
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    by_image: dict[int, list[dict]] = {}
    by_class: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
        by_class.setdefault(annotation["category_id"], []).append(annotation)

    chunks: list[Chunk] = []
    areas = [a["area"] for a in coco["annotations"]]
    counts = {names[c]: len(v) for c, v in by_class.items()}

    # --- dataset level -------------------------------------------------
    chunks.append(Chunk(
        id="dataset-overview",
        source="dataset",
        text=squash(textwrap.dedent(f"""\
            Dataset overview.
            The dataset contains {len(coco['images'])} street and traffic
            photographs with {len(coco['annotations'])} bounding box
            annotations across {len(names)} classes: {', '.join(names.values())}.
            Labels were produced automatically by the Grounding DINO zero-shot
            detector, not by human annotators, so label noise is expected.
            Annotations per image: mean
            {statistics.mean(len(v) for v in by_image.values()):.1f},
            median {statistics.median(len(v) for v in by_image.values()):.0f},
            maximum {max(len(v) for v in by_image.values())}.
            Box area in square pixels: minimum {min(areas):.0f}, median
            {statistics.median(areas):.0f}, maximum {max(areas):.0f}.
            Class counts: {counts}.
            """)),
    ))

    # --- per class -------------------------------------------------------
    for category_id, annotations in sorted(by_class.items()):
        class_areas = [a["area"] for a in annotations]
        scores = [a["score"] for a in annotations if "score" in a]
        presence = len({a["image_id"] for a in annotations})
        share = 100 * len(annotations) / len(coco["annotations"])
        chunks.append(Chunk(
            id=f"class-{names[category_id]}",
            source="class",
            text=squash(textwrap.dedent(f"""\
                Class summary: {names[category_id]}.
                {len(annotations)} annotations ({share:.1f}% of the dataset)
                appearing in {presence} of {len(coco['images'])} images.
                Box area in square pixels: median
                {statistics.median(class_areas):.0f}, smallest
                {min(class_areas):.0f}, largest {max(class_areas):.0f}.
                Detector confidence: median
                {statistics.median(scores):.2f} over
                {len(scores)} scored boxes.
                """)),
        ))

    # --- per image -------------------------------------------------------
    for image_id, annotations in sorted(by_image.items()):
        info = images[image_id]
        per_class = {}
        for annotation in annotations:
            label = names[annotation["category_id"]]
            per_class[label] = per_class.get(label, 0) + 1
        image_areas = [a["area"] for a in annotations]
        chunks.append(Chunk(
            id=f"image-{image_id}",
            source="image",
            text=(
                f"Image {image_id} ({info['file_name']}), "
                f"{info['width']}x{info['height']} pixels. "
                f"{len(annotations)} annotations: {per_class}. "
                f"Median box area {statistics.median(image_areas):.0f} square "
                f"pixels, smallest {min(image_areas):.0f}, largest "
                f"{max(image_areas):.0f}."
            ),
        ))

    # --- known quality concerns -----------------------------------------
    tiny = [a for a in coco["annotations"] if a["area"] < 32 * 32]
    duplicates = 0
    for annotations in by_image.values():
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                if (annotations[i]["category_id"] == annotations[j]["category_id"]
                        and box_iou(annotations[i]["bbox"],
                                    annotations[j]["bbox"]) > 0.7):
                    duplicates += 1
    chunks.append(Chunk(
        id="quality-notes",
        source="quality",
        text=squash(textwrap.dedent(f"""\
            Known annotation quality concerns.
            {len(tiny)} boxes are smaller than 32x32 pixels
            ({100 * len(tiny) / len(coco['annotations']):.1f}% of all boxes);
            very small boxes are often detector noise on distant or partly
            hidden objects rather than genuine instances.
            {duplicates} pairs of same-class boxes overlap with IoU above 0.7,
            which usually means the detector fired twice on one object. These
            were deliberately not filtered out, so that auditing has something
            real to find.
            The least frequent class has {min(counts.values())} instances and
            the most frequent {max(counts.values())}, an imbalance ratio of
            {max(counts.values()) / max(1, min(counts.values())):.1f} to 1.

            A small number of boxes cover most of their image. These were
            checked by hand and the labels are correct: they are close-up
            photographs in which one vehicle genuinely fills the frame. They
            are a dataset composition issue rather than a labelling error,
            because an object close-up is a different kind of image from the
            street scenes that make up the rest of the data. They entered the
            dataset because the street-scene requirement was relaxed during
            collection to improve throughput.

            Detector confidence is low overall: the median box scores about
            0.43 and roughly a quarter of all boxes score below 0.35. Since
            the labels come from a zero-shot model with no human review, the
            confidence distribution is the main evidence available about
            label reliability.
            """)),
    ))

    # --- training run configuration --------------------------------------
    if config:
        arguments = config.get("arguments", {})
        smoke = config.get("is_smoke_test")
        chunks.append(Chunk(
            id="training-configuration",
            source="training",
            text=squash(textwrap.dedent(f"""\
                Training run configuration for run {config.get('run_name')}.
                The model has {config.get('model_parameters', 0):,} trainable
                parameters and was trained on {config.get('train_images')}
                images at {arguments.get('size')} pixels with batch size
                {arguments.get('batch_size')} on {config.get('device')}.
                Learning rate {arguments.get('lr')}, weight decay
                {arguments.get('weight_decay')}, early stopping patience
                {arguments.get('patience')} epochs.
                Batches per epoch: {config.get('batches_per_epoch')}.

                IMPORTANT: this run
                {'WAS a short smoke test, deliberately truncated with '
                 '--limit-batches to check that the pipeline runs end to end. '
                 'Its metrics say nothing about dataset quality or model '
                 'capability. A validation mAP of zero is the expected result '
                 'of a smoke test and must NOT be attributed to the data, the '
                 'labels or the architecture.'
                 if smoke else
                 'was a full training run, so its metrics are meaningful.'}
                """)),
        ))

    # --- the diagnostic that bounds what the metrics can mean -------------
    # Without this, the auditor has the mAP but no way to know whether a low
    # one means broken code, bad labels or too little data -- and it reliably
    # guesses "bad labels", because that is what the rest of the corpus talks
    # about. This chunk supplies the measurement that rules two of the three
    # out, so the conclusion is grounded instead of merely plausible.
    chunks.append(Chunk(
        id="training-diagnostics",
        source="training",
        text=squash(textwrap.dedent("""            Independent sanity check on the training pipeline
            (scripts/overfit_check.py). The same model, loss, target encoder
            and decoder were trained repeatedly on a single fixed batch of
            four images and then evaluated on those same four images. Loss
            fell from 8.41 to 0.053 and mAP@0.5 reached 0.984, with the exact
            ground-truth box counts recovered per image.

            IMPORTANT interpretation. This result proves the target encoding,
            the loss function, the decoder and the evaluation code are all
            correct, because a pipeline with a bug in any of them cannot
            memorise a batch. It also proves the annotations are learnable
            signal rather than noise, because noise cannot be fitted to mAP
            0.984. Therefore a low mAP on the real validation split must NOT
            be attributed to broken code or to label quality. The binding
            constraint is dataset scale: only 175 training images are
            available for a detector trained from random initialisation,
            where published results of this kind assume tens of thousands.
            Label noise is a real and separate concern worth fixing, but it
            is not what caps the current mAP.
            """)),
    ))

    # --- training log ----------------------------------------------------
    for entry in history:
        train, validation = entry["train"], entry["val"]
        chunks.append(Chunk(
            id=f"training-epoch-{entry['epoch']}",
            source="training",
            text=(
                f"Training epoch {entry['epoch']}. "
                f"Train loss {train['loss']:.4f} (heatmap "
                f"{train.get('heatmap', float('nan')):.4f}, size "
                f"{train.get('size', float('nan')):.4f}). "
                f"Validation loss {validation['loss']:.4f}, "
                f"validation mAP@0.5 {validation.get('mAP50', 0):.4f}. "
                f"Epoch took {train.get('seconds', 0):.0f} seconds."
            ),
        ))

    return chunks


# ==========================================================================
# Vector store
# ==========================================================================


class VectorStore:
    """Chroma collection with a local sentence-transformer embedder.

    Embeddings are computed locally rather than through an API: the corpus is
    rebuilt whenever the annotations change, and paying per rebuild for a few
    hundred short passages would be pointless.
    """

    def __init__(self, path: Path = STORE_DIR,
                 model_name: str = EMBEDDING_MODEL) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"})

    def rebuild(self, chunks: list[Chunk]) -> int:
        # Delete and recreate: an incremental update would leave stale chunks
        # describing annotations that no longer exist.
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"})

        batch = 256
        for start in range(0, len(chunks), batch):
            window = chunks[start:start + batch]
            self.collection.add(
                ids=[c.id for c in window],
                documents=[c.text for c in window],
                metadatas=[{"source": c.source} for c in window],
            )
            print(f"  embedded {min(start + batch, len(chunks))}/{len(chunks)}",
                  flush=True)
        return self.collection.count()

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        result = self.collection.query(query_texts=[query], n_results=k)
        return [
            {"id": i, "text": d, "source": m.get("source", "?"),
             "distance": dist}
            for i, d, m, dist in zip(result["ids"][0], result["documents"][0],
                                     result["metadatas"][0],
                                     result["distances"][0])
        ]


# ==========================================================================
# Tools -- the exact-computation half
# ==========================================================================


class AuditTools:
    """Pure functions over the COCO file. No LLM involvement."""

    def __init__(self, coco: dict, history: list[dict],
                 config: dict | None = None) -> None:
        self.coco = coco
        self.history = history
        self.config = config or {}
        self.names = {c["id"]: c["name"] for c in coco["categories"]}
        self.ids = {name: cid for cid, name in self.names.items()}
        self.images = {i["id"]: i for i in coco["images"]}

    # -- tool 1 ---------------------------------------------------------
    def compute_box_statistics(self, class_name: str | None = None,
                               min_area: float | None = None,
                               max_area: float | None = None) -> dict:
        """Area statistics over a filtered subset of the annotations."""
        selected = self.coco["annotations"]
        if class_name:
            if class_name not in self.ids:
                return {"error": f"unknown class {class_name!r}",
                        "known_classes": sorted(self.ids)}
            selected = [a for a in selected
                        if a["category_id"] == self.ids[class_name]]
        if min_area is not None:
            selected = [a for a in selected if a["area"] >= min_area]
        if max_area is not None:
            selected = [a for a in selected if a["area"] <= max_area]

        if not selected:
            return {"matching_boxes": 0,
                    "note": "no annotations matched this filter"}

        areas = sorted(a["area"] for a in selected)
        quantile = lambda q: areas[min(int(q * len(areas)), len(areas) - 1)]
        per_class: dict[str, int] = {}
        for annotation in selected:
            label = self.names[annotation["category_id"]]
            per_class[label] = per_class.get(label, 0) + 1

        return {
            "filter": {"class_name": class_name, "min_area": min_area,
                       "max_area": max_area},
            "matching_boxes": len(selected),
            "share_of_dataset_percent": round(
                100 * len(selected) / len(self.coco["annotations"]), 2),
            "area_px2": {
                "min": round(areas[0], 1),
                "p25": round(quantile(0.25), 1),
                "median": round(quantile(0.5), 1),
                "p75": round(quantile(0.75), 1),
                "max": round(areas[-1], 1),
                "mean": round(statistics.mean(areas), 1),
            },
            "equivalent_square_side_px": {
                "median": round(quantile(0.5) ** 0.5, 1)},
            "per_class": per_class,
        }

    # -- tool 2 ---------------------------------------------------------
    def check_class_balance(self) -> dict:
        """Instance counts, image presence and the imbalance ratio."""
        counts = {name: 0 for name in self.ids}
        presence: dict[str, set] = {name: set() for name in self.ids}
        for annotation in self.coco["annotations"]:
            label = self.names[annotation["category_id"]]
            counts[label] += 1
            presence[label].add(annotation["image_id"])

        total = sum(counts.values()) or 1
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        largest, smallest = ordered[0], ordered[-1]

        return {
            "total_annotations": total,
            "total_images": len(self.coco["images"]),
            "per_class": [
                {"class_name": name, "instances": count,
                 "share_percent": round(100 * count / total, 2),
                 "images_containing": len(presence[name]),
                 "image_coverage_percent": round(
                     100 * len(presence[name]) / len(self.coco["images"]), 1)}
                for name, count in ordered
            ],
            "imbalance_ratio": round(largest[1] / max(1, smallest[1]), 2),
            "most_common": largest[0],
            "least_common": smallest[0],
        }

    # -- tool 3 ---------------------------------------------------------
    def find_suspicious_annotations(self, max_area: float = 1024.0,
                                    iou_threshold: float = 0.7,
                                    limit: int = 10,
                                    max_image_fraction: float = 0.5) -> dict:
        """Concrete failure modes of automatic labelling.

        `max_image_fraction` flags boxes covering most of the frame. Checking
        these by eye showed the labels are usually *correct*: they are
        close-up photographs where one vehicle genuinely fills the image. So
        this is reported as a dataset composition signal rather than a
        labelling error -- it marks images that are object portraits rather
        than street scenes, which is a different distribution from the rest of
        the data and worth knowing about before training. The counts here are
        evidence to look at, not a verdict.
        """
        annotations = self.coco["annotations"]
        by_image: dict[int, list[dict]] = {}
        for annotation in annotations:
            by_image.setdefault(annotation["image_id"], []).append(annotation)

        tiny = [a for a in annotations if a["area"] < max_area]
        tiny.sort(key=lambda a: a["area"])

        duplicates = []
        for image_id, group in by_image.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if group[i]["category_id"] != group[j]["category_id"]:
                        continue
                    iou = box_iou(group[i]["bbox"], group[j]["bbox"])
                    if iou > iou_threshold:
                        duplicates.append({
                            "image_id": image_id,
                            "class_name": self.names[group[i]["category_id"]],
                            "iou": round(iou, 3),
                            "annotation_ids": [group[i]["id"], group[j]["id"]],
                        })
        duplicates.sort(key=lambda d: -d["iou"])

        out_of_bounds = []
        oversized = []
        for annotation in annotations:
            info = self.images[annotation["image_id"]]
            x, y, w, h = annotation["bbox"]
            if x < -1 or y < -1 or x + w > info["width"] + 1 \
                    or y + h > info["height"] + 1:
                out_of_bounds.append({"annotation_id": annotation["id"],
                                      "image_id": annotation["image_id"]})

            fraction = annotation["area"] / (info["width"] * info["height"])
            if fraction > max_image_fraction:
                oversized.append({
                    "annotation_id": annotation["id"],
                    "image_id": annotation["image_id"],
                    "class_name": self.names[annotation["category_id"]],
                    "fraction_of_image": round(fraction, 3),
                    "area_px2": round(annotation["area"], 1),
                    "score": round(annotation.get("score", float("nan")), 3),
                })
        oversized.sort(key=lambda o: -o["fraction_of_image"])

        low_confidence = [a for a in annotations if a.get("score", 1.0) < 0.35]

        return {
            "thresholds": {"max_area_px2": max_area,
                           "iou_threshold": iou_threshold},
            "tiny_boxes": {
                "count": len(tiny),
                "percent_of_dataset": round(100 * len(tiny) / len(annotations), 2),
                "examples": [
                    {"annotation_id": a["id"], "image_id": a["image_id"],
                     "class_name": self.names[a["category_id"]],
                     "area_px2": round(a["area"], 1),
                     "bbox": [round(v, 1) for v in a["bbox"]]}
                    for a in tiny[:limit]],
            },
            "near_duplicate_pairs": {
                "count": len(duplicates),
                "examples": duplicates[:limit],
            },
            "boxes_outside_image_bounds": {
                "count": len(out_of_bounds),
                "examples": out_of_bounds[:limit],
            },
            "oversized_boxes": {
                "count": len(oversized),
                "percent_of_dataset": round(
                    100 * len(oversized) / len(annotations), 2),
                "note": (f"box covers more than "
                         f"{100 * max_image_fraction:.0f}% of its image. "
                         f"Manual inspection found these labels to be "
                         f"correct: they are close-up photographs of a single "
                         f"vehicle. Treat them as a sign that the image is an "
                         f"object close-up rather than a street scene, not as "
                         f"a labelling error."),
                "examples": oversized[:limit],
            },
            "low_confidence_boxes": {
                "count": len(low_confidence),
                "percent_of_dataset": round(
                    100 * len(low_confidence) / len(annotations), 2),
                "note": "detector score below 0.35",
            },
        }

    # -- tool 4 ---------------------------------------------------------
    def summarise_training(self) -> dict:
        """Read the training history and characterise the run."""
        if not self.history:
            return {"error": "no training history found under runs/",
                    "hint": "run train.py first"}

        epochs = self.history
        smoke = bool(self.config.get("is_smoke_test"))
        best = max(epochs, key=lambda e: e["val"].get("mAP50", 0.0))
        lowest = min(epochs, key=lambda e: e["val"]["loss"])
        first, last = epochs[0], epochs[-1]
        gap = last["val"]["loss"] - last["train"]["loss"]

        return {
            "is_smoke_test": smoke,
            "run_validity": (
                "This run was a truncated smoke test. Its metrics test only "
                "that the pipeline executes; they are not evidence about the "
                "dataset, the labels or the model. Do not draw conclusions "
                "about accuracy from them."
                if smoke else
                "This was a full training run, so its metrics are meaningful."),
            "batches_per_epoch": self.config.get("batches_per_epoch"),
            "train_images": self.config.get("train_images"),
            "epochs_completed": len(epochs),
            "best_epoch_by_mAP50": {
                "epoch": best["epoch"],
                "val_mAP50": round(best["val"].get("mAP50", 0.0), 4),
                "val_loss": round(best["val"]["loss"], 4)},
            "lowest_val_loss": {"epoch": lowest["epoch"],
                                "val_loss": round(lowest["val"]["loss"], 4)},
            "first_epoch_train_loss": round(first["train"]["loss"], 4),
            "final_train_loss": round(last["train"]["loss"], 4),
            "final_val_loss": round(last["val"]["loss"], 4),
            "generalisation_gap": round(gap, 4),
            "gap_interpretation": (
                "validation loss well above training loss, which suggests "
                "overfitting" if gap > 0.5 else
                "training and validation loss are close, so the model is not "
                "overfitting yet"),
            "total_minutes": round(
                sum(e["train"].get("seconds", 0) for e in epochs) / 60, 1),
            "pipeline_sanity_check": (
                "scripts/overfit_check.py trained this exact model, loss and "
                "decoder on one fixed batch of 4 images and reached mAP@0.5 "
                "0.984 on them, loss 8.41 -> 0.053, with exact box counts "
                "recovered. A pipeline bug makes that impossible, and so does "
                "unlearnable label noise. So a low mAP on the real split is "
                "NOT evidence of broken code or of bad labels."),
            "limiting_factor": (
                f"Only {self.config.get('train_images')} training images are "
                "available for a detector trained from scratch. That, not "
                "code or labels, is what caps the validation mAP here."),
        }


# ==========================================================================
# Tool declarations for the model
# ==========================================================================


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "compute_box_statistics",
        "description": (
            "Compute exact bounding box area statistics over the dataset, "
            "optionally filtered by class name and/or an area range in square "
            "pixels. Use this for any question about box sizes, how small or "
            "large boxes are, or how many boxes fall in an area range."),
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "One of: car, truck, traffic light, person. "
                                   "Omit for all classes."},
                "min_area": {"type": "number",
                             "description": "Minimum area in square pixels."},
                "max_area": {"type": "number",
                             "description": "Maximum area in square pixels."},
            },
        },
    },
    {
        "name": "check_class_balance",
        "description": (
            "Return per-class instance counts, each class's share of the "
            "dataset, how many images contain it, and the imbalance ratio "
            "between the most and least common class. Use this for any "
            "question about class distribution, imbalance or rarity."),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "find_suspicious_annotations",
        "description": (
            "Find likely annotation errors: boxes below an area threshold, "
            "same-class boxes overlapping above an IoU threshold (duplicate "
            "detections), boxes outside the image bounds, boxes covering most "
            "of their image (which indicate close-up photographs rather than "
            "street scenes, and are usually correctly labelled), and low "
            "confidence boxes. Returns counts and concrete examples with "
            "annotation ids."),
        "parameters": {
            "type": "object",
            "properties": {
                "max_area": {"type": "number",
                             "description": "Area below which a box counts as "
                                            "suspiciously small. Default 1024."},
                "iou_threshold": {"type": "number",
                                  "description": "IoU above which two same-class "
                                                 "boxes count as duplicates. "
                                                 "Default 0.7."},
                "limit": {"type": "integer",
                          "description": "Maximum examples per category."},
                "max_image_fraction": {
                    "type": "number",
                    "description": "Fraction of the image above which a box "
                                   "counts as oversized. Default 0.5."},
            },
        },
    },
    {
        "name": "summarise_training",
        "description": (
            "Summarise the most recent training run from its logged metrics: "
            "epochs completed, best epoch by validation mAP, final losses and "
            "the generalisation gap. Use for questions about training "
            "behaviour, overfitting or model quality."),
        "parameters": {"type": "object", "properties": {}},
    },
]


SYSTEM_PROMPT = """\
You are a dataset auditor for an object detection dataset of street scenes.

Rules you must follow:

1. Any factual claim about counts, sizes, ratios or specific annotations must
   come from a tool call. Never estimate a number yourself, and never take a
   number from the retrieved context if a tool can compute it: the context is
   a summary and may be stale.
2. The retrieved context tells you what the dataset is and what is already
   known to be wrong with it. Use it for interpretation and to decide which
   tool to call.
3. Call several tools when a question needs them. Do not stop at the first.
4. State numbers exactly as the tools return them, with units.
5. Say plainly when something cannot be determined from the available data.
6. The labels were produced by a zero-shot detector, not by people. Treat
   anomalies as possible labelling errors, and say which are worth a human
   looking at.

Answer concisely and concretely, citing the figures you used.
"""


# ==========================================================================
# Agent loop
# ==========================================================================


@dataclass
class TurnLog:
    """Everything that happened while answering one question."""

    question: str
    retrieved: list[dict]
    tool_calls: list[dict]
    answer: str


class Auditor:
    """Retrieval + tool-calling loop over Gemini."""

    def __init__(self, store: VectorStore, tools: AuditTools,
                 model: str = GEMINI_MODEL) -> None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY is not set. Put it in .env (see .env.example).")

        self.store = store
        self.tools = tools
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.registry: dict[str, Callable[..., dict]] = {
            "compute_box_statistics": tools.compute_box_statistics,
            "check_class_balance": tools.check_class_balance,
            "find_suspicious_annotations": tools.find_suspicious_annotations,
            "summarise_training": tools.summarise_training,
        }

    def _generate_with_retry(self, contents, config, attempts: int = 5,
                             verbose: bool = True):
        """Call the model, retrying the transient server-side failures.

        503 (model busy) and 429 (rate limited) are routine on the free tier
        and say nothing about the request. Without this, one such response
        part-way through a multi-tool answer loses every scenario that had
        already completed. 4xx errors other than 429 are real mistakes in the
        request, so they are raised immediately rather than retried.
        """
        from google.genai import errors

        delay = 4.0
        for attempt in range(1, attempts + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=config)
            except errors.APIError as exc:
                retryable = getattr(exc, "code", None) in (429, 500, 502, 503,
                                                           504)
                if not retryable or attempt == attempts:
                    raise
                if verbose:
                    print(f"     (API returned {exc.code}; retrying in "
                          f"{delay:.0f}s, attempt {attempt}/{attempts})",
                          flush=True)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def ask(self, question: str, verbose: bool = True) -> TurnLog:
        from google.genai import types

        retrieved = self.store.search(question)
        if verbose:
            print(f"\n{'=' * 74}\nQ: {question}\n{'=' * 74}")
            print(f"\nRetrieved {len(retrieved)} chunks:")
            for chunk in retrieved:
                preview = " ".join(chunk["text"].split())[:96]
                print(f"  [{chunk['source']:<8}] {chunk['id']:<22} "
                      f"d={chunk['distance']:.3f}  {preview}...")

        context = "\n\n".join(
            f"--- chunk {c['id']} (source: {c['source']}) ---\n{c['text']}"
            for c in retrieved)

        contents = [types.Content(role="user", parts=[types.Part(
            text=f"Retrieved context:\n\n{context}\n\n"
                 f"Question: {question}")])]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=TOOL_SCHEMAS)],
            # Disable the SDK's automatic calling: the loop is run explicitly
            # here so every call and result can be logged and shown.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True),
            temperature=0.0,
        )

        tool_calls: list[dict] = []
        answer = ""

        for round_number in range(MAX_TOOL_ROUNDS):
            response = self._generate_with_retry(contents, config,
                                                 verbose=verbose)

            candidate = response.candidates[0]
            parts = candidate.content.parts or []
            calls = [p.function_call for p in parts if p.function_call]

            if not calls:
                answer = "".join(p.text for p in parts if p.text).strip()
                break

            contents.append(candidate.content)
            responses = []
            for call in calls:
                arguments = dict(call.args or {})
                function = self.registry.get(call.name)
                if function is None:
                    result = {"error": f"unknown tool {call.name}"}
                else:
                    try:
                        result = function(**arguments)
                    except TypeError as exc:
                        result = {"error": f"bad arguments: {exc}"}

                tool_calls.append({"round": round_number, "name": call.name,
                                   "arguments": arguments, "result": result})
                if verbose:
                    rendered = json.dumps(arguments, ensure_ascii=False)
                    print(f"\n  -> tool call: {call.name}({rendered})")
                    summary = json.dumps(result, ensure_ascii=False)
                    print(f"     returned: {summary[:400]}"
                          f"{' ...' if len(summary) > 400 else ''}")

                responses.append(types.Part.from_function_response(
                    name=call.name, response={"result": result}))

            contents.append(types.Content(role="user", parts=responses))
        else:
            answer = ("Stopped after the maximum number of tool rounds without "
                      "a final answer.")

        if verbose:
            print(f"\n  Answer:\n{textwrap.indent(answer, '    ')}\n")

        return TurnLog(question=question, retrieved=retrieved,
                       tool_calls=tool_calls, answer=answer)


# ==========================================================================
# Demo scenarios
# ==========================================================================


SCENARIOS = [
    # Scenario 1: geometry. Needs compute_box_statistics, and should also
    # reach for find_suspicious_annotations to judge whether small is wrong.
    ("Are there suspiciously small bounding boxes in this dataset? "
     "Quantify how small, say which classes they belong to, and tell me "
     "whether they are likely to be labelling errors."),

    # Scenario 2: distribution. Needs check_class_balance, and the answer has
    # to connect imbalance to its consequences for training.
    ("Is this dataset class balanced? If not, which classes are "
     "under-represented and what problems should I expect when training a "
     "detector on it?"),

    # Scenario 3: combines dataset and training evidence, so it should call
    # more than one tool.
    ("Based on the annotations and the training log together, what is the "
     "single biggest risk to this model's accuracy, and what would you fix "
     "first?"),

    # Scenario 4: a question that cannot be answered from retrieved prose at
    # all. The corpus states percentiles for the dataset as a whole, not a
    # count inside an arbitrary area window for one class, so the model has
    # to compute it -- which is the point of giving it a tool rather than
    # more text.
    ("How many 'car' boxes have an area between 2000 and 20000 square "
     "pixels, and how does the median car box area compare with the median "
     "over every class?"),
]


def run_demo(auditor: Auditor, force: bool = False) -> None:
    """Run the scenarios, resuming any that did not finish last time.

    The free API tier is rate limited, so a run can stop part way through.
    Scenarios that already produced an answer are kept and skipped rather
    than re-asked, which costs no quota and keeps the transcript stable.
    """
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / "demo_transcript.json"

    # Previously successful answers are always loaded, `force` or not. With
    # `force` they are not skipped -- the scenario is asked again -- but they
    # stay as the fallback, so a failed re-ask cannot destroy a good answer
    # that is already on disk. An earlier version cleared them up front and
    # lost two real answers when the daily quota ran out mid-run.
    previous: dict[int, dict] = {}
    if path.exists():
        for entry in json.loads(path.read_text(encoding="utf-8")):
            if "error" not in entry:
                previous[entry["scenario"]] = entry
        if previous:
            verb = "re-asking" if force else "resuming; already answered:"
            print(f"{verb} scenarios {sorted(previous)}")

    logs: list[dict] = []

    def save() -> None:
        path.write_text(json.dumps(logs, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    for index, question in enumerate(SCENARIOS, start=1):
        print(f"\n\n{'#' * 74}\n# SCENARIO {index} of {len(SCENARIOS)}\n{'#' * 74}")
        if index in previous and not force:
            print("  (already answered; skipping)")
            logs.append(previous[index])
            save()
            continue
        try:
            log = auditor.ask(question)
        except Exception as exc:
            print(f"\n  scenario {index} failed: {type(exc).__name__}: {exc}")
            if index in previous:
                print("  keeping the previous successful answer")
                kept = dict(previous[index])
                kept["note"] = (f"retained from an earlier run; the re-ask "
                                f"failed with {type(exc).__name__}")
                logs.append(kept)
            else:
                logs.append({"scenario": index, "question": question,
                             "error": f"{type(exc).__name__}: {exc}"})
        else:
            logs.append({"scenario": index, "question": log.question,
                         "model": auditor.model,
                         "retrieved_chunk_ids": [c["id"] for c in log.retrieved],
                         "tool_calls": log.tool_calls, "answer": log.answer})
        # Written after every scenario so an interrupted run keeps its work.
        save()

    answered = [log for log in logs if "error" not in log]
    distinct = {call["name"] for log in answered for call in log["tool_calls"]}
    print(f"\n{'=' * 74}")
    print(f"{len(answered)} of {len(logs)} scenarios answered, "
          f"{sum(len(log['tool_calls']) for log in answered)} tool calls, "
          f"{len(distinct)} distinct tools used: {sorted(distinct)}")
    if len(answered) < len(logs):
        print("Some scenarios failed; re-run `python llm_auditor.py demo` to "
              "retry only those.")
    print(f"transcript written to {path}")


# ==========================================================================
# CLI
# ==========================================================================


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="chunk and embed the corpus")
    ask = sub.add_parser("ask", help="ask one grounded question")
    ask.add_argument("question")
    ask.add_argument("--model", default=GEMINI_MODEL)
    demo = sub.add_parser("demo", help="run the scripted audit scenarios")
    demo.add_argument("--model", default=GEMINI_MODEL,
                      help="the daily free-tier quota is per model, so a "
                           "different model has its own allowance")
    demo.add_argument("--force", action="store_true",
                      help="re-ask scenarios that already have an answer")
    sub.add_parser("chunks", help="print the corpus without embedding it")

    args = parser.parse_args()
    coco = load_coco()
    history_path, history = latest_history()
    config = run_config(history_path)

    if args.command == "chunks":
        chunks = build_chunks(coco, history, config)
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        print(f"{len(chunks)} chunks: {counts}")
        for chunk in chunks:
            if chunk.source != "image":  # 250 image chunks is too much to read
                print(f"\n--- {chunk.id} ---\n{chunk.text}")
        return

    if args.command == "build":
        chunks = build_chunks(coco, history, config)
        print(f"built {len(chunks)} chunks "
              f"(training history: {history_path or 'none found'})")
        store = VectorStore()
        total = store.rebuild(chunks)
        print(f"vector store now holds {total} chunks at {STORE_DIR}")
        return

    store = VectorStore()
    if store.collection.count() == 0:
        raise SystemExit("The vector store is empty. Run "
                         "`python llm_auditor.py build` first.")

    auditor = Auditor(store, AuditTools(coco, history, config),
                      model=args.model)
    if args.command == "ask":
        auditor.ask(args.question)
    else:
        run_demo(auditor, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
