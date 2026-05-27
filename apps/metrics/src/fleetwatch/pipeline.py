"""Windowing and orchestration: records -> per-window metric rows.

Groups detection records by (unit, window) and computes per-class metrics for
each group, plus per-(unit, class, condition) slices for each window. Metrics
are computed with the Python reference by default; pass ``use_cpp=True`` to route
through the C++ aggregator (they agree to 1e-9).
"""

from __future__ import annotations

from collections import defaultdict

from . import metrics_ref
from .aggregator import compute_cpp
from .metrics_ref import BatchMetrics
from .schema import DetectionRecord
from .store import SliceMetricRow, UnitMetricRow


def window_of(frame_id: int, frames_per_window: int) -> int:
    return frame_id // frames_per_window


def _compute(
    records: list[DetectionRecord], iou_threshold: float, use_cpp: bool
) -> BatchMetrics:
    if use_cpp:
        return compute_cpp(records, iou_threshold)
    return metrics_ref.compute(records, iou_threshold)


def build_rows(
    run_id: str,
    records: list[DetectionRecord],
    frames_per_window: int = 10,
    iou_threshold: float = 0.5,
    use_cpp: bool = False,
) -> tuple[list[UnitMetricRow], list[SliceMetricRow]]:
    """Compute per-(unit, class, window) metric rows and condition slices."""
    by_unit_window: dict[tuple[str, int], list[DetectionRecord]] = defaultdict(list)
    by_unit_window_cond: dict[tuple[str, int, str], list[DetectionRecord]] = defaultdict(list)

    for rec in records:
        w = window_of(rec.frame_id, frames_per_window)
        by_unit_window[(rec.unit_id, w)].append(rec)
        by_unit_window_cond[(rec.unit_id, w, rec.condition.slice_key())].append(rec)

    unit_rows: list[UnitMetricRow] = []
    for (unit_id, w), recs in sorted(by_unit_window.items()):
        m = _compute(recs, iou_threshold, use_cpp)
        for c in m.per_class:
            unit_rows.append(
                UnitMetricRow(
                    run_id=run_id,
                    unit_id=unit_id,
                    cls=c.cls,
                    window_idx=w,
                    tp=c.tp,
                    fp=c.fp,
                    fn=c.fn,
                    precision=c.precision,
                    recall=c.recall,
                    f1=c.f1,
                    ap=c.ap,
                )
            )

    slice_rows: list[SliceMetricRow] = []
    for (unit_id, w, cond), recs in sorted(by_unit_window_cond.items()):
        m = _compute(recs, iou_threshold, use_cpp)
        for c in m.per_class:
            slice_rows.append(
                SliceMetricRow(
                    run_id=run_id,
                    unit_id=unit_id,
                    cls=c.cls,
                    window_idx=w,
                    condition=cond,
                    precision=c.precision,
                    recall=c.recall,
                    ap=c.ap,
                )
            )

    return unit_rows, slice_rows
