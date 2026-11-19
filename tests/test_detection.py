"""Tests for the target encoder, the loss and the decoder.

The central test is `test_targets_round_trip_through_decode`: boxes are
encoded into heatmap/size/offset targets, those targets are fed to the
decoder as if they were a perfect prediction, and the recovered boxes are
compared with the originals. That closes the loop the training run depends
on -- if encoding and decoding disagree, the loss can fall to zero while the
detector outputs nonsense, and nothing else would catch it.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (  # noqa: E402
    STRIDE,
    CentreNet,
    DetectionLoss,
    EarlyStopping,
    WarmupCosine,
    average_precision,
    box_iou,
    build_targets,
    decode,
    focal_loss,
    gaussian_radius,
)

NUM_CLASSES = 4
INPUT = 384
OUTPUT = INPUT // STRIDE


# --------------------------------------------------------------------------
# Target encoding
# --------------------------------------------------------------------------


def test_heatmap_peaks_are_exactly_one_at_centres():
    boxes = [torch.tensor([[40.0, 60.0, 120.0, 140.0]])]
    labels = [torch.tensor([2])]
    targets = build_targets(boxes, labels, NUM_CLASSES, OUTPUT)

    heatmap = targets["heatmap"][0, 2]
    assert pytest.approx(1.0, abs=1e-6) == heatmap.max().item()
    # The peak belongs to class 2 only.
    for c in range(NUM_CLASSES):
        if c != 2:
            assert targets["heatmap"][0, c].max().item() == 0.0


def test_overlapping_objects_do_not_erase_each_other():
    """Two nearby same-class objects must both keep a peak of 1."""
    boxes = [torch.tensor([[40.0, 40.0, 120.0, 120.0],
                           [60.0, 60.0, 140.0, 140.0]])]
    labels = [torch.tensor([0, 0])]
    targets = build_targets(boxes, labels, NUM_CLASSES, OUTPUT)

    heatmap = targets["heatmap"][0, 0]
    assert int((heatmap == 1.0).sum()) == 2, "a Gaussian overwrote a peak"


def test_mask_marks_exactly_the_real_objects():
    boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
             torch.tensor([]).reshape(0, 4)]
    labels = [torch.tensor([1]), torch.tensor([], dtype=torch.long)]
    targets = build_targets(boxes, labels, NUM_CLASSES, OUTPUT)

    assert targets["mask"][0].sum() == 1
    assert targets["mask"][1].sum() == 0


def test_degenerate_boxes_are_skipped():
    boxes = [torch.tensor([[50.0, 50.0, 50.0, 90.0]])]  # zero width
    labels = [torch.tensor([0])]
    targets = build_targets(boxes, labels, NUM_CLASSES, OUTPUT)
    assert targets["mask"].sum() == 0


def test_gaussian_radius_grows_with_object_size():
    small = gaussian_radius(8, 8)
    large = gaussian_radius(64, 64)
    assert 0 <= small < large
    # A tolerance proportional to the box is the whole point of the formula.
    assert large > 4


# --------------------------------------------------------------------------
# The property that actually matters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("boxes_list", [
    [[50.0, 60.0, 150.0, 180.0]],
    [[20.0, 20.0, 60.0, 60.0], [200.0, 220.0, 300.0, 340.0]],
    [[8.0, 8.0, 28.0, 30.0]],            # a small object, the common case here
    [[4.0, 4.0, 372.0, 376.0]],          # nearly the whole frame
])
def test_targets_round_trip_through_decode(boxes_list):
    """Encode boxes, decode the targets as a perfect prediction, compare."""
    boxes = torch.tensor(boxes_list, dtype=torch.float32)
    labels = torch.zeros(len(boxes), dtype=torch.long)
    targets = build_targets([boxes], [labels], NUM_CLASSES, OUTPUT)

    # Rebuild dense size/offset maps from the sparse supervised entries, then
    # feed them to the decoder exactly as the model's outputs would be.
    size_map = torch.zeros(1, 2, OUTPUT, OUTPUT)
    offset_map = torch.zeros(1, 2, OUTPUT, OUTPUT)
    for k in range(int(targets["mask"][0].sum())):
        index = int(targets["index"][0, k])
        y, x = divmod(index, OUTPUT)
        size_map[0, :, y, x] = targets["size"][0, k]
        offset_map[0, :, y, x] = targets["offset"][0, k]

    # decode() applies a sigmoid, so invert it: a target of 1 must come back
    # as a confident logit, and everything else as a very negative one.
    logits = torch.full_like(targets["heatmap"], -20.0)
    logits[targets["heatmap"] == 1.0] = 20.0

    (recovered, recovered_labels, _), = decode(
        {"heatmap": logits, "size": size_map, "offset": offset_map},
        threshold=0.5,
    )

    assert len(recovered) == len(boxes), (
        f"encoded {len(boxes)} boxes, decoded {len(recovered)}")

    # Match by IoU rather than assuming decode preserves ordering.
    ious = box_iou(np.asarray(recovered), boxes.numpy())
    best = ious.max(axis=1)
    assert (best > 0.98).all(), f"round trip lost accuracy: IoU {best}"
    assert (recovered_labels == 0).all()


def test_decode_suppresses_non_peak_cells():
    """A smooth blob must yield one detection, not a cluster.

    This is what a trained heatmap actually looks like: a single maximum with
    values falling away from it. The 3x3 peak test should collapse the whole
    blob to its summit, which is the job NMS would otherwise do.
    """
    logits = torch.full((1, NUM_CLASSES, OUTPUT, OUTPUT), -20.0)
    ys = torch.arange(-3, 4, dtype=torch.float32)
    blob = 9.0 * torch.exp(-(ys[:, None] ** 2 + ys[None, :] ** 2) / 8.0)
    logits[0, 1, 39:46, 39:46] = blob
    size_map = torch.full((1, 2, OUTPUT, OUTPUT), 10.0)
    offset_map = torch.zeros(1, 2, OUTPUT, OUTPUT)

    (boxes, labels, scores), = decode(
        {"heatmap": logits, "size": size_map, "offset": offset_map},
        threshold=0.5)
    assert len(boxes) == 1, f"3x3 peak test let {len(boxes)} boxes through"
    assert labels[0] == 1


def test_exactly_flat_plateaus_are_a_known_limitation():
    """Documents a real property of max-pool NMS rather than hiding it.

    A cell survives when it *equals* its 3x3 maximum, so every cell of a
    perfectly flat plateau ties with its own neighbourhood and survives. This
    is inherited from the standard CenterNet peak test. It does not matter in
    practice -- exact ties between float sigmoid outputs are vanishingly
    unlikely -- but if a future change makes the heatmap quantised or
    saturating, this test is the warning that duplicate detections will
    follow.
    """
    logits = torch.full((1, NUM_CLASSES, OUTPUT, OUTPUT), -20.0)
    logits[0, 1, 40:44, 40:44] = 5.0  # a perfectly flat plateau
    size_map = torch.full((1, 2, OUTPUT, OUTPUT), 10.0)
    offset_map = torch.zeros(1, 2, OUTPUT, OUTPUT)

    (boxes, _, _), = decode(
        {"heatmap": logits, "size": size_map, "offset": offset_map},
        threshold=0.5)
    assert len(boxes) > 1, (
        "Flat plateaus now collapse to one peak -- the limitation this test "
        "documents has been fixed, so update the comment in decode()."
    )


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


def test_focal_loss_matches_closed_form_at_known_probability():
    """With p=0.01 everywhere and one positive, the loss must equal
    -log(0.01) * 0.99^2 plus a negligible negative term."""
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    logit = float(np.log(0.01 / 0.99))
    logits = torch.full((1, 1, 8, 8), logit)

    expected = -np.log(0.01) * (0.99 ** 2)
    assert focal_loss(logits, target).item() == pytest.approx(expected, rel=0.02)


def test_focal_loss_falls_as_prediction_improves():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 8, 8] = 1.0
    losses = [focal_loss(torch.where(target == 1.0,
                                     torch.full_like(target, logit),
                                     torch.full_like(target, -6.0)),
                         target).item()
              for logit in (-2.0, 0.0, 2.0, 6.0)]
    assert losses == sorted(losses, reverse=True), losses


def test_loss_is_finite_on_an_empty_image():
    """A batch with no objects must not produce NaN from a zero divide."""
    outputs = {
        "heatmap": torch.zeros(1, NUM_CLASSES, OUTPUT, OUTPUT),
        "size": torch.zeros(1, 2, OUTPUT, OUTPUT),
        "offset": torch.zeros(1, 2, OUTPUT, OUTPUT),
    }
    targets = build_targets([torch.zeros(0, 4)],
                            [torch.zeros(0, dtype=torch.long)],
                            NUM_CLASSES, OUTPUT)
    total, parts = DetectionLoss()(outputs, targets)
    assert torch.isfinite(total)
    assert all(np.isfinite(v) for v in parts.values())


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def test_model_output_shapes_are_stride_four():
    model = CentreNet(num_classes=NUM_CLASSES, width=8)
    outputs = model(torch.zeros(2, 3, INPUT, INPUT))
    assert outputs["heatmap"].shape == (2, NUM_CLASSES, OUTPUT, OUTPUT)
    assert outputs["size"].shape == (2, 2, OUTPUT, OUTPUT)
    assert outputs["offset"].shape == (2, 2, OUTPUT, OUTPUT)


def test_heatmap_starts_near_one_percent():
    """The bias initialisation must actually reach the output."""
    model = CentreNet(num_classes=NUM_CLASSES, width=8).eval()
    with torch.no_grad():
        probability = torch.sigmoid(
            model(torch.zeros(1, 3, 128, 128))["heatmap"]).mean().item()
    assert 0.003 < probability < 0.04, (
        f"heatmap starts at p={probability:.4f}; the head bias is being "
        "swamped by its weights")


def test_gradients_reach_the_stem():
    model = CentreNet(num_classes=NUM_CLASSES, width=8)
    boxes = [torch.tensor([[40.0, 40.0, 100.0, 100.0]])]
    labels = [torch.tensor([0])]
    targets = build_targets(boxes, labels, NUM_CLASSES, 128 // STRIDE)
    loss, _ = DetectionLoss()(model(torch.rand(1, 3, 128, 128)), targets)
    loss.backward()

    gradient = model.stem[0][0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0, "no gradient reached the first layer"


# --------------------------------------------------------------------------
# Schedule, stopping, metric
# --------------------------------------------------------------------------


def test_warmup_then_cosine_shape():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.0)
    scheduler = WarmupCosine(optimizer, warmup_steps=10, total_steps=100,
                             base_lr=1.0)
    rates = [scheduler.step() for _ in range(100)]

    assert rates[0] == pytest.approx(0.1)
    assert rates[9] == pytest.approx(1.0)        # peak at the end of warmup
    assert rates[:10] == sorted(rates[:10])      # rising
    assert rates[10:] == sorted(rates[10:], reverse=True)  # then falling
    assert rates[-1] < 0.01


def test_scheduler_does_not_go_negative_past_the_end():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.0)
    scheduler = WarmupCosine(optimizer, warmup_steps=2, total_steps=10,
                             base_lr=1.0)
    rates = [scheduler.step() for _ in range(40)]
    assert min(rates) >= 0.0


def test_early_stopping_counts_only_non_improvements():
    stopper = EarlyStopping(patience=3)
    assert stopper.update(0.10, 0) is True
    assert stopper.update(0.20, 1) is True
    assert not stopper.should_stop
    for epoch in range(2, 5):
        stopper.update(0.19, epoch)
    assert stopper.should_stop
    assert stopper.best == pytest.approx(0.20)
    assert stopper.best_epoch == 1


def test_falling_val_loss_counts_as_progress_while_map_is_noise():
    """Guards the bug that stopped a healthy run at epoch 13.

    mAP peaked at 0.005 by chance and never beat it, while validation loss
    improved every epoch. Monitoring mAP alone stopped a model that was
    plainly still learning.
    """
    stopper = EarlyStopping(patience=3, map_floor=0.02)
    stopper.update(0.0051, 0, val_loss=4.4)
    for epoch, loss in enumerate([4.2, 4.0, 3.9, 3.8, 3.7], start=1):
        stopper.update(0.001, epoch, val_loss=loss)
        assert not stopper.should_stop, (
            f"stopped at epoch {epoch} while validation loss was still "
            f"falling")


def test_stops_once_both_map_and_loss_stall():
    stopper = EarlyStopping(patience=3, map_floor=0.02)
    stopper.update(0.001, 0, val_loss=4.0)
    for epoch in range(1, 4):
        stopper.update(0.001, epoch, val_loss=4.0)
    assert stopper.should_stop


def test_map_alone_decides_once_it_is_meaningful():
    """Above the floor, a falling loss must not keep a stalled run alive."""
    stopper = EarlyStopping(patience=2, map_floor=0.02)
    stopper.update(0.30, 0, val_loss=4.0)
    stopper.update(0.29, 1, val_loss=3.0)
    stopper.update(0.28, 2, val_loss=2.0)
    assert stopper.should_stop, "loss improvement overrode a meaningful mAP"


def test_average_precision_is_one_for_a_perfect_ranking():
    records = [(0.9, True), (0.8, True), (0.7, True)]
    assert average_precision(records, positives=3) == pytest.approx(1.0)


def test_average_precision_penalises_false_positives_ranked_first():
    good = average_precision([(0.9, True), (0.5, False)], positives=1)
    bad = average_precision([(0.9, False), (0.5, True)], positives=1)
    assert good > bad


def test_average_precision_is_zero_without_ground_truth():
    assert average_precision([(0.9, False)], positives=0) == 0.0
