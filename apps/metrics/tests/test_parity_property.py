"""Property-based parity: C++ aggregator == Python reference on random batches."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fleetwatch import metrics_ref
from fleetwatch.aggregator import aggregator_path, compute_cpp
from fleetwatch.schema import DetectionRecord

TOL = 1e-9
CLASSES = ["car", "pedestrian", "cyclist", "sign"]

pytestmark = pytest.mark.skipif(aggregator_path() is None, reason="aggregator not built")


@st.composite
def boxes(draw: st.DrawFn) -> list[float]:
    x1 = draw(st.floats(min_value=0, max_value=500))
    y1 = draw(st.floats(min_value=0, max_value=500))
    w = draw(st.floats(min_value=1, max_value=200))
    h = draw(st.floats(min_value=1, max_value=200))
    return [x1, y1, x1 + w, y1 + h]


@st.composite
def records(draw: st.DrawFn) -> DetectionRecord:
    n_det = draw(st.integers(min_value=0, max_value=6))
    n_gt = draw(st.integers(min_value=0, max_value=6))
    dets = [
        {
            "class": draw(st.sampled_from(CLASSES)),
            "bbox": draw(boxes()),
            "confidence": draw(st.floats(min_value=0.0, max_value=1.0)),
        }
        for _ in range(n_det)
    ]
    gts = [
        {"class": draw(st.sampled_from(CLASSES)), "bbox": draw(boxes())} for _ in range(n_gt)
    ]
    return DetectionRecord.model_validate(
        {
            "unit_id": "u",
            "frame_id": 0,
            "timestamp": 0.0,
            "condition": {"lighting": "day", "weather": "clear", "distance_band": "near"},
            "detections": dets,
            "ground_truth": gts,
        }
    )


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=120, deadline=None)
@given(batch=st.lists(records(), min_size=1, max_size=10))
def test_cpp_equals_python_reference(batch: list[DetectionRecord]) -> None:
    ref = metrics_ref.compute(batch, 0.5)
    cpp = compute_cpp(batch, 0.5)
    assert cpp.map == pytest.approx(ref.map, abs=TOL)
    assert cpp.micro_precision == pytest.approx(ref.micro_precision, abs=TOL)
    assert cpp.micro_recall == pytest.approx(ref.micro_recall, abs=TOL)
    assert len(cpp.per_class) == len(ref.per_class)
    for cc, rc in zip(cpp.per_class, ref.per_class, strict=True):
        assert cc.cls == rc.cls
        assert cc.tp == rc.tp
        assert cc.fp == rc.fp
        assert cc.fn == rc.fn
        assert cc.precision == pytest.approx(rc.precision, abs=TOL)
        assert cc.recall == pytest.approx(rc.recall, abs=TOL)
        assert cc.f1 == pytest.approx(rc.f1, abs=TOL)
        assert cc.ap == pytest.approx(rc.ap, abs=TOL)
