"""View-model layer for the dashboard, computed from metric rows.

Kept independent of FastAPI and of Postgres so it can be unit-tested directly on
the rows produced by ``pipeline.build_rows``. The API binds these view models to
templates; an end-to-end run can build them from an in-memory pipeline result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .alerts import Alert, evaluate
from .attribution import Attribution, attribute
from .changepoint import ChangePoint, ShiftKind, detect
from .drift import compute_drift
from .store import SliceMetricRow, UnitMetricRow


@dataclass
class UnitSummary:
    unit_id: str
    mean_recall: float
    mean_precision: float
    worst_class: str
    worst_recall: float


@dataclass
class DriftRow:
    unit_id: str
    cls: str
    slope: float
    delta: float
    drifting: bool
    series: list[tuple[int, float]]
    change_kind: str = ShiftKind.NONE.value
    change_window: int | None = None
    change_magnitude: float = 0.0


@dataclass
class HeatCell:
    unit_id: str
    condition: str
    metric_value: float
    delta: float  # late-minus-early change of the metric for this cell


@dataclass
class DashboardModel:
    run_id: str
    units: list[UnitSummary] = field(default_factory=list)
    drift_rows: list[DriftRow] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    heatmap: list[HeatCell] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    attributions: list[Attribution] = field(default_factory=list)


def _series(
    rows: list[UnitMetricRow], unit_id: str, cls: str, metric: str
) -> list[tuple[int, float]]:
    pts = [
        (r.window_idx, getattr(r, metric))
        for r in rows
        if r.unit_id == unit_id and r.cls == cls
    ]
    return sorted(pts)


def build_model(
    run_id: str,
    unit_rows: list[UnitMetricRow],
    slice_rows: list[SliceMetricRow],
    metric: str = "recall",
    drop_threshold: float = 0.10,
) -> DashboardModel:
    model = DashboardModel(run_id=run_id)

    # Per-unit summary across all of its classes.
    by_unit: dict[str, list[UnitMetricRow]] = defaultdict(list)
    for r in unit_rows:
        by_unit[r.unit_id].append(r)
    for unit_id, rows in sorted(by_unit.items()):
        recalls = [r.recall for r in rows]
        precisions = [r.precision for r in rows]
        worst = min(rows, key=lambda r: r.recall)
        model.units.append(
            UnitSummary(
                unit_id=unit_id,
                mean_recall=sum(recalls) / len(recalls),
                mean_precision=sum(precisions) / len(precisions),
                worst_class=worst.cls,
                worst_recall=worst.recall,
            )
        )

    # Drift per (unit, class) and the alerts derived from it.
    pairs = sorted({(r.unit_id, r.cls) for r in unit_rows})
    for unit_id, cls in pairs:
        series = _series(unit_rows, unit_id, cls, metric)
        drift = compute_drift(unit_id, cls, series, metric, drop_threshold)
        cp: ChangePoint = detect(series, threshold=drop_threshold)
        model.drift_rows.append(
            DriftRow(
                unit_id=unit_id,
                cls=cls,
                slope=drift.slope,
                delta=drift.delta,
                drifting=drift.drifting,
                series=series,
                change_kind=cp.kind.value,
                change_window=cp.window,
                change_magnitude=cp.magnitude,
            )
        )
        # Attribute the degradation to the driving condition axis value, and use
        # it as the alert's worst-condition instead of a plain min over slices.
        attr = attribute(unit_id, cls, slice_rows, metric)
        if drift.drifting:
            model.attributions.append(attr)
        if drift.drifting and attr.driver_axis is not None:
            worst_cond: str | None = f"{attr.driver_axis}={attr.driver_value}"
            worst_val: float | None = attr.ranked[0].late_mean
        else:
            worst_cond, worst_val = _worst_condition(slice_rows, unit_id, cls, metric)
        alert = evaluate(
            drift,
            worst_condition=worst_cond if drift.drifting else None,
            worst_condition_value=worst_val if drift.drifting else None,
        )
        if alert is not None:
            model.alerts.append(alert)

    # Condition heatmap: mean metric per (unit, condition) over all classes,
    # plus the late-minus-early delta that colours the attribution heatmap.
    model.conditions = sorted({sr.condition for sr in slice_rows})
    cell_acc: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for sr in slice_rows:
        cell_acc[(sr.unit_id, sr.condition)].append((sr.window_idx, getattr(sr, metric)))
    for (unit_id, cond), pts in sorted(cell_acc.items()):
        vals = [v for _, v in pts]
        model.heatmap.append(
            HeatCell(
                unit_id=unit_id,
                condition=cond,
                metric_value=sum(vals) / len(vals),
                delta=_cell_delta(pts),
            )
        )

    return model


def _cell_delta(points: list[tuple[int, float]]) -> float:
    """Late-segment mean minus early-segment mean for one heatmap cell."""
    ordered = sorted(points)
    if len(ordered) < 2:
        return 0.0
    half = max(1, len(ordered) // 3)
    early = [v for _, v in ordered[:half]]
    late = [v for _, v in ordered[-half:]]
    return sum(late) / len(late) - sum(early) / len(early)


def _worst_condition(
    slice_rows: list[SliceMetricRow], unit_id: str, cls: str, metric: str
) -> tuple[str | None, float | None]:
    by_cond: dict[str, list[float]] = defaultdict(list)
    for r in slice_rows:
        if r.unit_id == unit_id and r.cls == cls:
            by_cond[r.condition].append(getattr(r, metric))
    if not by_cond:
        return None, None
    means = {cond: sum(v) / len(v) for cond, v in by_cond.items()}
    worst = min(means, key=lambda c: means[c])
    return worst, means[worst]
