"""Hand-computed unit tests for the Python reference metrics."""

from __future__ import annotations

import pytest

from fleetwatch.metrics_ref import average_precision, compute, iou
from fleetwatch.schema import Condition, DetectionRecord, DistanceBand, Lighting, Weather

COND = Condition(lighting=Lighting.DAY, weather=Weather.CLEAR, distance_band=DistanceBand.NEAR)


def rec(dets: list[dict], gts: list[dict], frame: int = 0) -> DetectionRecord:
    return DetectionRecord.model_validate(
        {
            "unit_id": "u",
            "frame_id": frame,
            "timestamp": float(frame),
            "condition": {"lighting": "day", "weather": "clear", "distance_band": "near"},
            "detections": dets,
            "ground_truth": gts,
        }
    )


def test_iou_identical_is_one() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap() -> None:
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-12)


def test_iou_symmetric() -> None:
    assert iou((1, 2, 9, 7), (3, 1, 12, 8)) == iou((3, 1, 12, 8), (1, 2, 9, 7))


def test_perfect_frame_map_one() -> None:
    r = rec(
        [
            {"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9},
            {"class": "pedestrian", "bbox": [20, 20, 30, 30], "confidence": 0.8},
        ],
        [
            {"class": "car", "bbox": [0, 0, 10, 10]},
            {"class": "pedestrian", "bbox": [20, 20, 30, 30]},
        ],
    )
    m = compute([r], 0.5)
    assert m.map == pytest.approx(1.0)
    assert m.micro_precision == pytest.approx(1.0)
    assert m.micro_recall == pytest.approx(1.0)


def test_missed_gt_is_false_negative() -> None:
    r = rec([], [{"class": "car", "bbox": [0, 0, 10, 10]}])
    m = compute([r], 0.5)
    assert m.per_class[0].tp == 0
    assert m.per_class[0].fn == 1
    assert m.per_class[0].recall == 0.0


def test_class_mismatch_is_false_positive() -> None:
    r = rec(
        [{"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9}],
        [{"class": "pedestrian", "bbox": [0, 0, 10, 10]}],
    )
    m = compute([r], 0.5)
    car = next(c for c in m.per_class if c.cls == "car")
    assert car.fp == 1
    assert car.tp == 0


def test_higher_confidence_claims_gt_first() -> None:
    r = rec(
        [
            {"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.6},
            {"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9},
        ],
        [{"class": "car", "bbox": [0, 0, 10, 10]}],
    )
    m = compute([r], 0.5)
    car = next(c for c in m.per_class if c.cls == "car")
    assert car.tp == 1
    assert car.fp == 1


def test_average_precision_hand_computed() -> None:
    # 2 GT, detections TP, FP, TP -> AP = 0.8333333... (see C++ test_prcurve).
    from fleetwatch.metrics_ref import ScoredDet, _pr_points

    scored = [
        ScoredDet(0.9, True, "c"),
        ScoredDet(0.8, False, "c"),
        ScoredDet(0.7, True, "c"),
    ]
    ap = average_precision(_pr_points(scored, 2), 2)
    assert ap == pytest.approx(0.8333333333333333, abs=1e-9)


def test_two_class_map_averages() -> None:
    r = rec(
        [
            {"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9},
            {"class": "pedestrian", "bbox": [100, 100, 110, 110], "confidence": 0.7},
        ],
        [
            {"class": "car", "bbox": [0, 0, 10, 10]},
            {"class": "pedestrian", "bbox": [0, 0, 10, 10]},
        ],
    )
    m = compute([r], 0.5)
    assert m.map == pytest.approx(0.5)
