"""Property tests for metric invariants using Hypothesis."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fleetwatch.metrics_ref import compute, iou
from fleetwatch.schema import Condition, DetectionRecord, DistanceBand, Lighting, Weather

CLASSES = ["car", "pedestrian", "cyclist", "sign"]
COND = Condition(lighting=Lighting.DAY, weather=Weather.CLEAR, distance_band=DistanceBand.NEAR)


@st.composite
def boxes(draw: st.DrawFn) -> tuple[float, float, float, float]:
    x1 = draw(st.floats(min_value=0, max_value=500))
    y1 = draw(st.floats(min_value=0, max_value=500))
    w = draw(st.floats(min_value=1, max_value=200))
    h = draw(st.floats(min_value=1, max_value=200))
    return (x1, y1, x1 + w, y1 + h)


@st.composite
def detections(draw: st.DrawFn) -> dict:
    return {
        "class": draw(st.sampled_from(CLASSES)),
        "bbox": list(draw(boxes())),
        "confidence": draw(st.floats(min_value=0.0, max_value=1.0)),
    }


@st.composite
def ground_truths(draw: st.DrawFn) -> dict:
    return {"class": draw(st.sampled_from(CLASSES)), "bbox": list(draw(boxes()))}


@st.composite
def records(draw: st.DrawFn) -> DetectionRecord:
    dets = draw(st.lists(detections(), max_size=8))
    gts = draw(st.lists(ground_truths(), max_size=8))
    return DetectionRecord.model_validate(
        {
            "unit_id": "u",
            "frame_id": draw(st.integers(min_value=0, max_value=1000)),
            "timestamp": 0.0,
            "condition": {"lighting": "day", "weather": "clear", "distance_band": "near"},
            "detections": dets,
            "ground_truth": gts,
        }
    )


@given(a=boxes(), b=boxes())
def test_iou_in_unit_interval(a, b) -> None:
    v = iou(a, b)
    assert 0.0 <= v <= 1.0


@given(a=boxes(), b=boxes())
def test_iou_symmetric(a, b) -> None:
    assert iou(a, b) == iou(b, a)


@given(a=boxes())
def test_iou_self_is_one(a) -> None:
    assert iou(a, a) == 1.0


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=200)
@given(batch=st.lists(records(), min_size=1, max_size=12))
def test_precision_recall_in_unit_interval(batch: list[DetectionRecord]) -> None:
    m = compute(batch, 0.5)
    for c in m.per_class:
        assert 0.0 <= c.precision <= 1.0
        assert 0.0 <= c.recall <= 1.0
        assert 0.0 <= c.f1 <= 1.0
        assert 0.0 <= c.ap <= 1.0
    assert 0.0 <= m.map <= 1.0


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
@given(gts=st.lists(ground_truths(), min_size=1, max_size=8))
def test_perfect_detections_are_precision_recall_one(gts: list[dict]) -> None:
    # Reproduce every GT box as a detection -> precision = recall = 1.
    dets = [{"class": g["class"], "bbox": g["bbox"], "confidence": 0.9} for g in gts]
    rec = DetectionRecord.model_validate(
        {
            "unit_id": "u",
            "frame_id": 0,
            "timestamp": 0.0,
            "condition": {"lighting": "day", "weather": "clear", "distance_band": "near"},
            "detections": dets,
            "ground_truth": gts,
        }
    )
    m = compute([rec], 0.5)
    assert m.micro_precision == 1.0
    assert m.micro_recall == 1.0
    assert m.map == 1.0


@settings(max_examples=100)
@given(gts=st.lists(ground_truths(), min_size=1, max_size=8))
def test_empty_detections_give_zero_recall(gts: list[dict]) -> None:
    rec = DetectionRecord.model_validate(
        {
            "unit_id": "u",
            "frame_id": 0,
            "timestamp": 0.0,
            "condition": {"lighting": "day", "weather": "clear", "distance_band": "near"},
            "detections": [],
            "ground_truth": gts,
        }
    )
    m = compute([rec], 0.5)
    assert m.micro_recall == 0.0
    for c in m.per_class:
        assert c.recall == 0.0
        assert c.tp == 0
