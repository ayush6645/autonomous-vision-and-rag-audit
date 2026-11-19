"""Collect and auto-annotate a street-scene image dataset.

Stage 1 (this module): scrape high-resolution images from Wikimedia Commons
with Playwright, verify each one with Gemini 2.5 Flash Lite, and record a
reproducible manifest.

Stage 2 (added later): zero-shot annotation with Grounding DINO -> COCO JSON.

Usage:
    python collect_and_annotate.py scrape --target 400
    python collect_and_annotate.py scrape --target 20 --no-verify
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import requests
from dotenv import load_dotenv
from PIL import Image

# Playwright is imported lazily inside the scrape stage: the annotate and
# validate stages run on machines (Kaggle, Colab) with no browser installed.

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
REJECT_DIR = DATA_DIR / "rejected"
CACHE_PATH = DATA_DIR / "verdict_cache.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"
COCO_PATH = DATA_DIR / "annotations.json"
LOG_PATH = ROOT / "logs" / "scrape.log"

SEARCH_URL = (
    "https://commons.wikimedia.org/w/index.php"
    "?search={term}&title=Special:MediaSearch&type=image&continue={offset}"
)
RESULT_SELECTOR = "a.sdms-image-result"
PAGE_SIZE = 40  # MediaSearch returns 40 hits per `continue` offset
MAX_PAGES = 4

# MediaSearch returns few hits for long, specific phrases, so prefer short
# queries and breadth across cities rather than depth on any one term.
# Ordered by measured yield: candidates are consumed in this order and the
# run stops once the target is met, so the most productive terms come first.
# Percentages are the observed accept rate from the first full run.
SEARCH_TERMS = [
    # Measured high yield.
    "traffic",              # 86%
    "pedestrian crossing",  # 76%
    "urban traffic",        # 55%
    "street scene",         # 52%
    "city street",          # 49%
    # Untested, but phrased like the winners above.
    "traffic jam", "traffic light", "crosswalk pedestrians",
    "city road vehicles", "road intersection", "busy street",
    "urban intersection", "highway traffic", "city bus street",
    "taxi street", "truck road", "downtown street",
    "street corner city", "main street town", "city centre street",
    "avenue city", "boulevard traffic", "street photography",
    "street view city", "roundabout",
    # Dropped: "street traffic cars" (31%), "car park street" and
    # "street market city" returned mostly object-free scenes.
]

TARGET_CLASSES = ["car", "truck", "traffic light", "person"]
MIN_SIDE = 800
MIN_INSTANCES = 2
MAX_BYTES = 20 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Only motion blur is disqualifying: it destroys object boundaries, which is
# exactly what the detector must learn. Night and watermark images are usable
# and add real-world variety. `is_photograph` stays strict, because
# illustrations and 3D renders do not resemble the target domain at all.
BLOCKING_ISSUES = {"blur"}

# Whether the scene must be a street. Relaxed: the detector is trained on the
# four target classes, and a car park or plaza contains them just as well.
REQUIRE_STREET_SCENE = False

DOWNLOAD_WORKERS = 4  # parallel fetches; Wikimedia tolerates this comfortably
CHUNK_SIZE = 12  # candidates downloaded per batch before verification

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_MIN_INTERVAL = 4.5  # seconds between calls, free-tier RPM headroom

# Wikimedia's user-agent policy requires a descriptive agent with a contact
# route; generic agents are throttled with HTTP 429.
USER_AGENT = (
    "AugleAI-DatasetCollector/1.0 "
    "(https://github.com/augle-ai-assessment; contact via repo issues)"
)
DOWNLOAD_DELAY = 0.15  # polite gap between downloads
VERIFY_MAX_SIDE = 1024  # originals are downscaled before upload to the VLM

# Zero-shot annotation. Grounding DINO expects lowercase, period-separated
# phrases; "tiny" is used because the pipeline must run on CPU.
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DINO_PROMPT = ". ".join(TARGET_CLASSES) + "."
MIN_BOX_SIDE = 2.0  # pixels; anything thinner is a degenerate detection

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_photograph": {"type": "boolean"},
        "is_street_scene": {"type": "boolean"},
        "visible_classes": {"type": "array", "items": {"type": "string"}},
        "instance_estimate": {"type": "integer"},
        "quality_issues": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": [
        "is_photograph",
        "is_street_scene",
        "visible_classes",
        "instance_estimate",
        "quality_issues",
        "reason",
    ],
}

VERIFY_PROMPT = f"""You are a dataset quality gate for an object-detection project.
The target classes are: {', '.join(TARGET_CLASSES)}.

Judge ONLY this image. Do not draw boxes. Report:
- is_photograph: true for a real photograph; false for illustration, 3D render,
  diagram, map, screenshot or clipart.
- is_street_scene: true if it shows an outdoor road, street or traffic scene.
- visible_classes: which target classes are clearly visible.
- instance_estimate: total count of target-class objects you can see.
- quality_issues: any of blur, night, low_light, watermark, occluded, interior.
- reason: one short sentence.

Answer with JSON only."""

LOGGER = logging.getLogger("collect")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    """A search hit before it has been downloaded."""

    source_page_url: str
    image_url: str
    title: str
    search_term: str


@dataclass
class Record:
    """One image's full provenance, written to the manifest."""

    id: int
    filename: str
    sha256: str
    width: int
    height: int
    bytes: int
    image_format: str
    source_page_url: str
    image_url: str
    title: str
    search_term: str
    license: str
    downloaded_at: str
    status: str  # accepted | rejected
    reject_reason: str | None = None
    verification: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


def load_json(path: Path, default: Any) -> Any:
    """Read JSON, returning *default* when the file is absent or corrupt."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Corrupt JSON at %s; starting fresh.", path)
        return default


def save_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so an interrupted run never truncates the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def configure_logging(verbose: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


# --------------------------------------------------------------------------
# Stage 1a: search
# --------------------------------------------------------------------------


def original_url(thumb_url: str) -> str | None:
    """Derive the full-resolution Commons URL from a thumbnail URL.

    Commons thumbnails follow a fixed pattern::

        https://<host>/wikipedia/commons/thumb/6/6c/NAME.jpg/330px-NAME.jpg

    The original lives at::

        https://upload.wikimedia.org/wikipedia/commons/6/6c/NAME.jpg

    Returns None when the URL does not match that pattern.
    """
    match = re.search(r"/wikipedia/commons/thumb/(\w/\w\w/[^/]+)/", thumb_url)
    if not match:
        return None
    return f"https://upload.wikimedia.org/wikipedia/commons/{match.group(1)}"


def search_commons(page, term: str, target: int) -> list[Candidate]:
    """Collect up to *target* candidates for one search term."""
    from playwright.sync_api import Error as PlaywrightError

    candidates: list[Candidate] = []
    pages = min(MAX_PAGES, -(-target // PAGE_SIZE))

    for page_index in range(pages):
        url = SEARCH_URL.format(
            term=requests.utils.quote(term), offset=page_index * PAGE_SIZE
        )
        page.goto(url, timeout=60_000)
        try:
            page.wait_for_selector(RESULT_SELECTOR, timeout=15_000)
        except PlaywrightError:
            break  # no results on this offset: term exhausted

        # Results below the fold lazy-load their <img>, so scroll the grid
        # before extracting or the thumbnail URLs are still placeholders.
        for _ in range(5):
            page.mouse.wheel(0, 20_000)
            page.wait_for_timeout(300)

        # Extract in one evaluate() call: per-element get_attribute() blocks
        # for the full timeout whenever an image has not loaded yet.
        rows = page.evaluate(
            """(sel) => [...document.querySelectorAll(sel)].map((a) => ({
                href: a.getAttribute('href') || '',
                src: a.querySelector('img')?.getAttribute('src') || '',
            }))""",
            RESULT_SELECTOR,
        )

        for row in rows:
            full = original_url(row["src"])
            if not full or not row["href"]:
                continue
            candidates.append(
                Candidate(
                    source_page_url=row["href"],
                    image_url=full,
                    title=unquote(full.rsplit("/", 1)[-1]),
                    search_term=term,
                )
            )
        if len(rows) < PAGE_SIZE:
            break  # final page for this term

    LOGGER.info("%-28s -> %3d candidates", term, len(candidates))
    return candidates[:target]


# --------------------------------------------------------------------------
# Stage 1b: download and decode
# --------------------------------------------------------------------------


def download(url: str, dest: Path, retries: int = 3) -> int | None:
    """Download *url* to *dest*. Returns the byte count, or None on failure."""
    headers = {"User-Agent": USER_AGENT, "Accept": "image/*"}
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
                if resp.status_code == 429:
                    wait = int(resp.headers.get("retry-after", 10 * attempt))
                    LOGGER.warning("Rate limited; sleeping %ds.", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                size = int(resp.headers.get("content-length", 0))
                if size > MAX_BYTES:
                    LOGGER.debug("Too large (%d bytes): %s", size, url)
                    return None
                dest.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with dest.open("wb") as handle:
                    for chunk in resp.iter_content(65_536):
                        written += len(chunk)
                        if written > MAX_BYTES:
                            dest.unlink(missing_ok=True)
                            return None
                        handle.write(chunk)
                time.sleep(DOWNLOAD_DELAY)
                return written
        except requests.RequestException as exc:
            LOGGER.debug("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            time.sleep(2**attempt)
    return None


def inspect_image(path: Path) -> tuple[int, int, str] | None:
    """Decode the file to confirm it is a usable image of sufficient size."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format or ""
    except Exception:  # noqa: BLE001 - any decode failure is a reject
        return None
    if fmt not in ALLOWED_FORMATS:
        return None
    if min(width, height) < MIN_SIDE:
        return None
    return width, height, fmt


def encode_for_vlm(path: Path, max_side: int = VERIFY_MAX_SIDE) -> str:
    """Downscale to a bounded JPEG and return it base64-encoded.

    Quality judgements do not need full resolution, and uploading 10 MB
    originals would be slow and risk the inline-payload limit. Converting to
    JPEG also keeps the declared mime type honest for PNG and WEBP inputs.
    """
    buffer = io.BytesIO()
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def safe_filename(title: str, index: int) -> str:
    """Build a filesystem-safe, collision-free name from a Commons title.

    Commons titles may contain quotes, colons and other characters that are
    illegal on Windows, so keep only a conservative character set and prefix
    a zero-padded index to guarantee uniqueness.
    """
    stem, _, suffix = title.rpartition(".")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem or title).strip("_")[:60]
    suffix = re.sub(r"[^A-Za-z0-9]", "", suffix).lower() or "jpg"
    return f"{index:05d}_{stem}.{suffix}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Stage 1c: Gemini quality gate
# --------------------------------------------------------------------------


class GeminiVerifier:
    """Thin REST client with a verdict cache and free-tier rate limiting."""

    def __init__(self, api_key: str, cache_path: Path = CACHE_PATH) -> None:
        self.api_key = api_key
        self.cache_path = cache_path
        self.cache: dict[str, dict[str, Any]] = load_json(cache_path, {})
        self._last_call = 0.0

    def verify(self, path: Path, digest: str) -> dict[str, Any]:
        """Return a verdict dict, reusing the cached one when available."""
        if digest in self.cache:
            return {**self.cache[digest], "cached": True}

        elapsed = time.monotonic() - self._last_call
        if elapsed < GEMINI_MIN_INTERVAL:
            time.sleep(GEMINI_MIN_INTERVAL - elapsed)

        payload = {
            "contents": [{
                "parts": [
                    {"text": VERIFY_PROMPT},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": encode_for_vlm(path),
                    }},
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": VERDICT_SCHEMA,
            },
        }
        self._last_call = time.monotonic()
        try:
            # Key travels in a header, never the URL, so it cannot leak
            # into logs or exception messages.
            resp = requests.post(
                GEMINI_URL,
                headers={"x-goog-api-key": self.api_key},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            verdict = json.loads(text)
        except (requests.RequestException, KeyError, IndexError,
                json.JSONDecodeError) as exc:
            LOGGER.warning("Verification failed for %s: %s", path.name, exc)
            return {"error": str(exc), "cached": False}

        verdict["model"] = GEMINI_MODEL
        self.cache[digest] = verdict
        save_json(self.cache_path, self.cache)
        return {**verdict, "cached": False}


def is_usable(verdict: dict[str, Any]) -> tuple[bool, str | None]:
    """Apply the acceptance policy to a Gemini verdict."""
    if "error" in verdict:
        return False, "verification_error"
    if not verdict.get("is_photograph"):
        return False, "not_a_photograph"
    if REQUIRE_STREET_SCENE and not verdict.get("is_street_scene"):
        return False, "not_a_street_scene"
    if verdict.get("instance_estimate", 0) < MIN_INSTANCES:
        return False, "too_few_objects"
    issues = BLOCKING_ISSUES & set(verdict.get("quality_issues", []))
    if issues:
        return False, f"quality:{','.join(sorted(issues))}"
    return True, None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def gather_candidates(terms: Iterable[str], per_term: int) -> list[Candidate]:
    """Run the Playwright search across every term, de-duplicating by URL."""
    from playwright.sync_api import sync_playwright

    seen: set[str] = set()
    collected: list[Candidate] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        for term in terms:
            try:
                hits = search_commons(page, term, per_term)
            except Exception as exc:  # noqa: BLE001 - one bad term must not stop the run
                LOGGER.warning("Search failed for %r: %s", term, exc)
                continue
            for hit in hits:
                if hit.image_url not in seen:
                    seen.add(hit.image_url)
                    collected.append(hit)
        browser.close()
    return collected


def resume_records() -> list[Record]:
    """Load records from a previous run so no download or quota is repeated.

    A record is only kept when its image file is still on disk, so a manually
    cleaned data directory is reflected correctly.
    """
    manifest = load_json(MANIFEST_PATH, None)
    if not manifest:
        return []
    known = {field.name for field in dataclasses.fields(Record)}
    records: list[Record] = []
    for entry in manifest.get("images", []):
        folder = IMAGE_DIR if entry.get("status") == "accepted" else REJECT_DIR
        if not (folder / entry.get("filename", "")).exists():
            continue
        records.append(Record(**{k: v for k, v in entry.items() if k in known}))
    return records


def run_scrape(args: argparse.Namespace) -> None:
    """Funnel: search -> download -> decode -> verify -> manifest."""
    started = datetime.now(timezone.utc).isoformat()
    # Observed yield is roughly 40% of candidates (sub-800px originals and
    # VLM rejects), so over-provision the candidate pool by ~4x.
    per_term = max(40, args.target // len(SEARCH_TERMS) * 4)
    candidates = gather_candidates(SEARCH_TERMS, per_term)
    LOGGER.info("Collected %d unique candidates.", len(candidates))

    verifier = None
    if not args.no_verify:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            LOGGER.error("GEMINI_API_KEY is not set. Use --no-verify to skip.")
            sys.exit(1)
        verifier = GeminiVerifier(api_key)

    # Resume: reuse everything a previous run already downloaded and verified.
    records = resume_records()
    hashes = {record.sha256 for record in records}
    seen_urls = {record.image_url for record in records}
    next_id = max((record.id for record in records), default=0) + 1
    stats = {
        "candidates": len(candidates),
        "downloaded": len(records),
        "decoded": len(records),
        "duplicates": 0,
        "accepted": sum(r.status == "accepted" for r in records),
        "rejected": sum(r.status == "rejected" for r in records),
        "resumed": len(records),
    }
    if records:
        LOGGER.info("Resuming: %d already processed (%d accepted).",
                    len(records), stats["accepted"])

    pending = [c for c in candidates if c.image_url not in seen_urls]
    LOGGER.info("%d new candidates to process.", len(pending))

    for start in range(0, len(pending), CHUNK_SIZE):
        if stats["accepted"] >= args.target:
            break
        chunk = pending[start:start + CHUNK_SIZE]

        # Downloads are I/O-bound and dominate the runtime, so fetch a batch
        # in parallel. Verification stays sequential: it is rate-limited.
        jobs = [(candidate, IMAGE_DIR / safe_filename(candidate.title, next_id + i))
                for i, candidate in enumerate(chunk)]
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            sizes = list(pool.map(
                lambda job: download(job[0].image_url, job[1]), jobs
            ))

        for (candidate, staging), size in zip(jobs, sizes):
            if stats["accepted"] >= args.target:
                break
            if size is None:
                continue
            stats["downloaded"] += 1

            probe = inspect_image(staging)
            if probe is None:
                staging.unlink(missing_ok=True)
                continue
            width, height, fmt = probe
            stats["decoded"] += 1

            digest = sha256_of(staging)
            if digest in hashes:
                staging.unlink(missing_ok=True)
                stats["duplicates"] += 1
                continue
            hashes.add(digest)

            if verifier is None:
                verdict, accepted, reason = {"skipped": True}, True, None
            else:
                verdict = verifier.verify(staging, digest)
                accepted, reason = is_usable(verdict)

            # Keep the on-disk name aligned with the record id.
            final = (IMAGE_DIR if accepted else REJECT_DIR) / safe_filename(
                candidate.title, next_id)
            final.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(final)
            stats["accepted" if accepted else "rejected"] += 1

            records.append(Record(
                id=next_id,
                filename=final.name,
                sha256=digest,
                width=width,
                height=height,
                bytes=size,
                image_format=fmt,
                source_page_url=candidate.source_page_url,
                image_url=candidate.image_url,
                title=candidate.title,
                search_term=candidate.search_term,
                license="Wikimedia Commons free license (see source_page_url)",
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                status="accepted" if accepted else "rejected",
                reject_reason=reason,
                verification=verdict,
            ))
            next_id += 1

        # The target can be reached mid-chunk, leaving files that were
        # downloaded but never recorded. Drop them so the image directory
        # and the manifest always agree.
        keep = {folder / record.filename
                for record in records
                for folder in (IMAGE_DIR, REJECT_DIR)}
        for _, staging in jobs:
            if staging.exists() and staging not in keep:
                staging.unlink(missing_ok=True)

        _write_manifest(records, stats, started, args)
        LOGGER.info("Progress: %(accepted)d accepted / %(rejected)d rejected",
                    stats)

    _write_manifest(records, stats, started, args)
    LOGGER.info("Done. %s", json.dumps(stats))
    LOGGER.info("Manifest: %s", MANIFEST_PATH)


def _write_manifest(records: list[Record], stats: dict[str, int],
                    started: str, args: argparse.Namespace) -> None:
    save_json(MANIFEST_PATH, {
        "run": {
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "target": args.target,
            "search_terms": SEARCH_TERMS,
            "target_classes": TARGET_CLASSES,
            "min_side": MIN_SIDE,
            "min_instances": MIN_INSTANCES,
            "verifier": None if args.no_verify else GEMINI_MODEL,
            "funnel": stats,
        },
        "images": [asdict(record) for record in records],
    })


# --------------------------------------------------------------------------
# Stage 2: zero-shot annotation -> COCO JSON
# --------------------------------------------------------------------------


def load_detector(device: str):
    """Load Grounding DINO once and reuse it across the dataset."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    LOGGER.info("Loading %s on %s ...", DINO_MODEL, device)
    processor = AutoProcessor.from_pretrained(DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL)
    model.to(device).eval()
    return processor, model


def detect(path: Path, processor, model, device: str, box_threshold: float,
           text_threshold: float, shortest_edge: int = 800) -> list[dict[str, Any]]:
    """Run zero-shot detection and return boxes in absolute xyxy pixels.

    `shortest_edge` controls the tensor the backbone sees. The default of 800
    matches Grounding DINO's training resolution; lowering it is the single
    most effective CPU speed-up, at a modest cost in small-object recall.
    Boxes are always returned in the ORIGINAL image's pixel space.
    """
    import torch

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        width, height = image.size

    inputs = processor(
        images=image,
        text=DINO_PROMPT,
        return_tensors="pt",
        size={"shortest_edge": shortest_edge,
              "longest_edge": int(shortest_edge * 1.35)},
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(height, width)],
    )[0]

    # Older transformers returns "labels", newer returns "text_labels".
    labels = results.get("text_labels", results.get("labels"))
    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], labels):
        category = match_category(str(label))
        if category is None:
            continue
        x_min, y_min, x_max, y_max = (float(v) for v in box.tolist())
        detections.append({
            "category": category,
            "score": round(float(score), 4),
            "xyxy": (x_min, y_min, x_max, y_max),
        })
    return detections


def match_category(label: str) -> str | None:
    """Map a free-text Grounding DINO label onto a target class.

    The model emits fragments of the prompt ("traffic", "light", "a car"),
    so match on substring containment in both directions rather than equality.
    """
    text = label.lower().strip()
    if not text:
        return None
    for name in TARGET_CLASSES:
        if name in text or text in name:
            return name
    # "traffic" alone most often refers to the traffic light phrase.
    if text in {"traffic", "light"}:
        return "traffic light"
    return None


def clamp_box(xyxy: tuple[float, float, float, float], width: int,
              height: int) -> tuple[float, float, float, float] | None:
    """Clip a box to the image and reject degenerate results."""
    x_min, y_min, x_max, y_max = xyxy
    x_min, y_min = max(0.0, x_min), max(0.0, y_min)
    x_max, y_max = min(float(width), x_max), min(float(height), y_max)
    if x_max - x_min < MIN_BOX_SIDE or y_max - y_min < MIN_BOX_SIDE:
        return None
    return x_min, y_min, x_max, y_max


def build_coco(entries: list[tuple[dict[str, Any], list[dict[str, Any]]]],
               thresholds: dict[str, float]) -> dict[str, Any]:
    """Assemble a COCO-format dictionary from per-image detections."""
    categories = [
        {"id": index, "name": name, "supercategory": "street"}
        for index, name in enumerate(TARGET_CLASSES, start=1)
    ]
    category_ids = {category["name"]: category["id"] for category in categories}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for record, detections in entries:
        images.append({
            "id": record["id"],
            "file_name": record["filename"],
            "width": record["width"],
            "height": record["height"],
            "license": 1,
            "coco_url": record["image_url"],
            "flickr_url": record["source_page_url"],
            "date_captured": record["downloaded_at"],
        })
        for detection in detections:
            box = clamp_box(detection["xyxy"], record["width"], record["height"])
            if box is None:
                continue
            x_min, y_min, x_max, y_max = box
            # Round the box first, then derive area from the rounded values,
            # so `area` and `bbox` can never disagree downstream.
            x, y = round(x_min, 2), round(y_min, 2)
            w, h = round(x_max - x_min, 2), round(y_max - y_min, 2)
            annotations.append({
                "id": annotation_id,
                "image_id": record["id"],
                "category_id": category_ids[detection["category"]],
                "bbox": [x, y, w, h],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "segmentation": [],
                "score": detection["score"],
            })
            annotation_id += 1

    return {
        "info": {
            "description": "Street-scene dataset, zero-shot annotated.",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "contributor": "Augle AI intern assessment",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "annotation_model": DINO_MODEL,
            "thresholds": thresholds,
        },
        "licenses": [{
            "id": 1,
            "name": "Wikimedia Commons free license",
            "url": "https://commons.wikimedia.org/wiki/Commons:Licensing",
        }],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def validate_coco(path: Path) -> dict[str, Any]:
    """Check schema compliance and geometric sanity. Raises on violation."""
    from pycocotools.coco import COCO

    data = load_json(path, None)
    if data is None:
        raise ValueError(f"No COCO file at {path}")

    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"COCO file is missing the '{key}' block.")

    image_ids = {image["id"] for image in data["images"]}
    category_ids = {category["id"] for category in data["categories"]}
    if len(image_ids) != len(data["images"]):
        raise ValueError("Duplicate image ids.")

    sizes = {image["id"]: (image["width"], image["height"])
             for image in data["images"]}
    seen_annotation_ids: set[int] = set()

    for annotation in data["annotations"]:
        ann_id = annotation["id"]
        if ann_id in seen_annotation_ids:
            raise ValueError(f"Duplicate annotation id {ann_id}.")
        seen_annotation_ids.add(ann_id)
        if annotation["image_id"] not in image_ids:
            raise ValueError(f"Annotation {ann_id} references a missing image.")
        if annotation["category_id"] not in category_ids:
            raise ValueError(f"Annotation {ann_id} has an unknown category.")

        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0:
            raise ValueError(f"Annotation {ann_id} has a non-positive side.")
        if x < 0 or y < 0:
            raise ValueError(f"Annotation {ann_id} starts outside the image.")
        img_w, img_h = sizes[annotation["image_id"]]
        if x + w > img_w + 1 or y + h > img_h + 1:
            raise ValueError(f"Annotation {ann_id} extends past the image.")
        if abs(annotation["area"] - w * h) > 1.0:
            raise ValueError(f"Annotation {ann_id} area disagrees with bbox.")

    # Final authority: pycocotools must parse the file without complaint.
    COCO(str(path))

    counts = collections.Counter(
        annotation["category_id"] for annotation in data["annotations"]
    )
    names = {category["id"]: category["name"]
             for category in data["categories"]}
    return {
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "per_class": {names[cid]: counts.get(cid, 0) for cid in names},
        "images_without_annotations": len(
            image_ids - {a["image_id"] for a in data["annotations"]}
        ),
    }


def run_annotate(args: argparse.Namespace) -> None:
    """Annotate every accepted image and write COCO JSON."""
    manifest = load_json(MANIFEST_PATH, None)
    if manifest is None:
        LOGGER.error("No manifest at %s. Run the scrape command first.",
                     MANIFEST_PATH)
        sys.exit(1)

    records = [r for r in manifest["images"] if r["status"] == "accepted"]
    if args.limit:
        records = records[: args.limit]
    LOGGER.info("Annotating %d images.", len(records))

    processor, model = load_detector(args.device)
    thresholds = {"box": args.box_threshold, "text": args.text_threshold}
    entries: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    for position, record in enumerate(records, start=1):
        path = IMAGE_DIR / record["filename"]
        if not path.exists():
            LOGGER.warning("Missing file, skipping: %s", record["filename"])
            continue
        try:
            detections = detect(path, processor, model, args.device,
                                args.box_threshold, args.text_threshold,
                                args.shortest_edge)
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop the run
            LOGGER.warning("Detection failed for %s: %s", path.name, exc)
            continue
        entries.append((record, detections))
        if position % 10 == 0 or position == len(records):
            LOGGER.info("  %d/%d images, %d boxes so far", position,
                        len(records), sum(len(d) for _, d in entries))

    save_json(COCO_PATH, build_coco(entries, thresholds))
    summary = validate_coco(COCO_PATH)
    LOGGER.info("COCO written to %s", COCO_PATH)
    LOGGER.info("Validation passed: %s", json.dumps(summary))


def run_validate(args: argparse.Namespace) -> None:
    """Validate an existing COCO file without re-running detection."""
    LOGGER.info("Validation passed: %s",
                json.dumps(validate_coco(COCO_PATH), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scrape = sub.add_parser("scrape", help="collect and verify images")
    scrape.add_argument("--target", type=int, default=400,
                        help="number of ACCEPTED images to collect")
    scrape.add_argument("--no-verify", action="store_true",
                        help="skip the Gemini quality gate")
    scrape.add_argument("--verbose", action="store_true")
    scrape.set_defaults(func=run_scrape)

    annotate = sub.add_parser("annotate", help="zero-shot annotate -> COCO")
    annotate.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    annotate.add_argument("--box-threshold", type=float, default=0.30)
    annotate.add_argument("--text-threshold", type=float, default=0.25)
    annotate.add_argument("--shortest-edge", type=int, default=800,
                          help="backbone input size; lower is much faster on CPU")
    annotate.add_argument("--limit", type=int, default=0,
                          help="annotate only the first N images (testing)")
    annotate.add_argument("--verbose", action="store_true")
    annotate.set_defaults(func=run_annotate)

    validate = sub.add_parser("validate", help="validate the COCO file")
    validate.add_argument("--verbose", action="store_true")
    validate.set_defaults(func=run_validate)

    args = parser.parse_args()
    configure_logging(getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
