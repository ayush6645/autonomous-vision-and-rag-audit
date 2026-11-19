"""Geometry tests for the augmentation pipeline.

The important test here is `test_boxes_agree_with_warped_pixels`. Checking a
transformed box against hand-computed numbers only proves the arithmetic
matches itself. Instead each box is rendered as a mask, the mask is warped
with the *same* matrix used on the image, and the bounds of the warped mask
are compared with the predicted box. That is the property the training loop
actually depends on: pixels and labels moving together.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import (  # noqa: E402
    GeometricPipeline,
    Sample,
    clip_and_filter,
    horizontal_flip,
    identity,
    rotation,
    scaling,
    shear,
    transform_boxes,
    translation,
)

CANVAS = 400


def make_sample(boxes: np.ndarray) -> Sample:
    image = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    return Sample(
        image=image,
        boxes=boxes.astype(np.float32),
        labels=np.zeros(len(boxes), dtype=np.int64),
        image_id=0,
    )


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_identity_leaves_boxes_untouched():
    boxes = np.array([[10, 20, 110, 220]], dtype=np.float32)
    assert np.allclose(transform_boxes(boxes, identity()), boxes)


def test_translation_shifts_exactly():
    boxes = np.array([[10, 20, 110, 220]], dtype=np.float32)
    moved = transform_boxes(boxes, translation(15, -5))
    assert np.allclose(moved, [[25, 15, 125, 215]])


def test_scaling_scales_exactly():
    boxes = np.array([[10, 20, 110, 220]], dtype=np.float32)
    scaled = transform_boxes(boxes, scaling(2.0, 0.5))
    assert np.allclose(scaled, [[20, 10, 220, 110]])


def test_horizontal_flip_mirrors():
    boxes = np.array([[10, 20, 110, 220]], dtype=np.float32)
    flipped = transform_boxes(boxes, horizontal_flip(CANVAS))
    # x1' = W - x2, x2' = W - x1; y is unchanged.
    assert np.allclose(flipped, [[CANVAS - 110, 20, CANVAS - 10, 220]])


def test_flip_is_an_involution():
    boxes = np.array([[10, 20, 110, 220], [300, 5, 390, 99]], dtype=np.float32)
    once = transform_boxes(boxes, horizontal_flip(CANVAS))
    twice = transform_boxes(once, horizontal_flip(CANVAS))
    assert np.allclose(twice, boxes)


def test_rotation_by_90_degrees_is_exact():
    """A square box rotated 90 degrees about the canvas centre."""
    boxes = np.array([[100, 150, 200, 250]], dtype=np.float32)
    matrix = rotation(90.0, (CANVAS / 2, CANVAS / 2))
    rotated = transform_boxes(boxes, matrix)

    cx = cy = CANVAS / 2
    corners = [(100, 150), (200, 150), (200, 250), (100, 250)]
    expected = [(cx - (y - cy), cy + (x - cx)) for x, y in corners]
    xs = [p[0] for p in expected]
    ys = [p[1] for p in expected]
    assert np.allclose(rotated, [[min(xs), min(ys), max(xs), max(ys)]], atol=1e-4)


def test_round_trip_through_inverse():
    boxes = np.array([[30, 40, 170, 260]], dtype=np.float32)
    matrix = (rotation(23.0, (200, 200))
              @ shear(5.0, -3.0, (200, 200))
              @ scaling(1.3, 0.8))
    back = transform_boxes(transform_boxes(boxes, matrix),
                           np.linalg.inv(matrix))
    # Rotation grows the axis-aligned hull, so the round trip is not the
    # identity on the box: it must *contain* the original, never shrink it.
    assert back[0, 0] <= boxes[0, 0] + 1e-3
    assert back[0, 1] <= boxes[0, 1] + 1e-3
    assert back[0, 2] >= boxes[0, 2] - 1e-3
    assert back[0, 3] >= boxes[0, 3] - 1e-3


# --------------------------------------------------------------------------
# The property that actually matters
# --------------------------------------------------------------------------


MATRICES = {
    "identity": identity(),
    "translate": translation(37, -21),
    "scale_up": scaling(1.4, 1.4),
    "scale_down": scaling(0.6, 0.75),
    "rotate_15": rotation(15.0, (CANVAS / 2, CANVAS / 2)),
    "rotate_-32": rotation(-32.0, (CANVAS / 2, CANVAS / 2)),
    "shear": shear(9.0, -6.0, (CANVAS / 2, CANVAS / 2)),
    "flip": horizontal_flip(CANVAS),
    "composed": (translation(12, 9)
                 @ rotation(11.0, (CANVAS / 2, CANVAS / 2))
                 @ shear(4.0, 3.0, (CANVAS / 2, CANVAS / 2))
                 @ scaling(0.9, 1.1)
                 @ horizontal_flip(CANVAS)),
}


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_boxes_agree_with_warped_pixels(name: str):
    """Predicted boxes must match the bounds of the warped mask."""
    matrix = MATRICES[name]
    boxes = np.array([
        [40.0, 60.0, 180.0, 170.0],
        [210.0, 100.0, 300.0, 280.0],
        [150.0, 220.0, 190.0, 260.0],
    ], dtype=np.float32)

    predicted = transform_boxes(boxes, matrix)

    for index, box in enumerate(boxes):
        mask = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
        x1, y1, x2, y2 = box.astype(int)
        mask[y1:y2, x1:x2] = 255

        warped = cv2.warpAffine(
            mask, matrix[:2], (CANVAS, CANVAS),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        ys, xs = np.nonzero(warped)
        if len(xs) == 0:
            continue  # warped entirely off-canvas; nothing to compare

        actual = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                          dtype=np.float32)

        # The mask can only occupy the canvas, so a box whose true extent
        # runs past the edge is reported by the pixels as stopping at the
        # edge. transform_boxes deliberately does NOT clip -- that is
        # clip_and_filter's job -- so clip the prediction before comparing.
        expected = predicted[index].copy()
        expected[[0, 1]] = expected[[0, 1]].clip(0, CANVAS)
        expected[[2, 3]] = expected[[2, 3]].clip(0, CANVAS)

        # Tolerance covers nearest-neighbour rasterisation of the mask edge.
        assert np.allclose(expected, actual, atol=2.0), (
            f"{name}: box {index} predicted {expected} "
            f"but pixels landed at {actual}"
        )


def test_two_corner_shortcut_is_wrong_under_rotation():
    """Guards against the classic bug of warping only two corners.

    If someone 'simplifies' transform_boxes to map just the top-left and
    bottom-right points, this test should start failing.
    """
    boxes = np.array([[40.0, 60.0, 180.0, 170.0]], dtype=np.float32)
    matrix = rotation(30.0, (CANVAS / 2, CANVAS / 2))

    correct = transform_boxes(boxes, matrix)

    points = np.array([[40.0, 60.0, 1.0], [180.0, 170.0, 1.0]])
    warped = points @ matrix.T
    naive = np.array([[warped[:, 0].min(), warped[:, 1].min(),
                       warped[:, 0].max(), warped[:, 1].max()]])

    assert not np.allclose(correct, naive, atol=1.0), (
        "The two-corner shortcut agreed with the four-corner result, "
        "which means the four-corner logic is not being exercised."
    )


# --------------------------------------------------------------------------
# Clipping and filtering
# --------------------------------------------------------------------------


def test_clip_keeps_boxes_inside_canvas():
    boxes = np.array([[-50, -20, 100, 120], [350, 350, 500, 500]],
                     dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    clipped, kept = clip_and_filter(boxes, labels, CANVAS, CANVAS)

    assert (clipped[:, 0] >= 0).all() and (clipped[:, 1] >= 0).all()
    assert (clipped[:, 2] <= CANVAS).all() and (clipped[:, 3] <= CANVAS).all()
    assert len(kept) == len(clipped)


def test_barely_visible_boxes_are_dropped():
    """A box pushed almost entirely off-canvas should not survive."""
    boxes = np.array([[-95.0, 10.0, 5.0, 110.0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    original_areas = np.array([100.0 * 100.0], dtype=np.float32)

    _, kept = clip_and_filter(boxes, labels, CANVAS, CANVAS, original_areas)
    assert len(kept) == 0  # only 5% of the area remains


def test_degenerate_boxes_are_dropped():
    boxes = np.array([[10.0, 10.0, 10.5, 200.0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    _, kept = clip_and_filter(boxes, labels, CANVAS, CANVAS)
    assert len(kept) == 0


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


def test_pipeline_output_is_square_and_in_bounds():
    import random

    boxes = np.array([[40.0, 60.0, 180.0, 170.0],
                      [210.0, 100.0, 300.0, 280.0]], dtype=np.float32)
    pipeline = GeometricPipeline(size=256, train=True)
    rng = random.Random(0)

    for _ in range(200):
        out = pipeline(make_sample(boxes), rng)
        assert out.image.shape == (256, 256, 3)
        assert len(out.boxes) == len(out.labels)
        if len(out.boxes):
            assert (out.boxes[:, 0] >= 0).all()
            assert (out.boxes[:, 1] >= 0).all()
            assert (out.boxes[:, 2] <= 256).all()
            assert (out.boxes[:, 3] <= 256).all()
            assert (out.boxes[:, 2] > out.boxes[:, 0]).all()
            assert (out.boxes[:, 3] > out.boxes[:, 1]).all()


def test_eval_pipeline_is_deterministic_and_lossless_in_aspect():
    import random

    boxes = np.array([[0.0, 0.0, 400.0, 400.0]], dtype=np.float32)
    pipeline = GeometricPipeline(size=256, train=False)
    first = pipeline(make_sample(boxes), random.Random(1))
    second = pipeline(make_sample(boxes), random.Random(99))

    assert np.allclose(first.boxes, second.boxes)
    # A full-canvas square box should map to the whole 256 canvas.
    assert np.allclose(first.boxes, [[0, 0, 256, 256]], atol=1.0)
