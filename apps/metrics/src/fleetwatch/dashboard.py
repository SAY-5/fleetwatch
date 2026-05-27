"""View-model layer for the dashboard, computed from metric rows.

Kept independent of FastAPI and of Postgres so it can be unit-tested directly on
the rows produced by ``pipeline.build_rows``. The API binds these view models to
templates; an end-to-end run can build them from an in-memory pipeline result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .alerts import Alert, evaluate
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


@dataclass
class HeatCell:
    unit_id: str
    condition: str
    metric_value: float


@dataclass
class DashboardModel:
    run_id: str
    units: list[UnitSummary] = field(default_factory=list)
    drift_rows: list[DriftRow] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    heatmap: list[HeatCell] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)


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
        model.drift_rows.append(
            DriftRow(
                unit_id=unit_id,
                cls=cls,
                slope=drift.slope,
                delta=drift.delta,
                drifting=drift.drifting,
                series=series,
            )
        )
        worst_cond, worst_val = _worst_condition(slice_rows, unit_id, cls, metric)
        alert = evaluate(
            drift,
            worst_condition=worst_cond if drift.drifting else None,
            worst_condition_value=worst_val if drift.drifting else None,
        )
        if alert is not None:
            model.alerts.append(alert)

    # Condition heatmap: mean metric per (unit, condition) over all classes.
    model.conditions = sorted({sr.condition for sr in slice_rows})
    cell_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sr in slice_rows:
        cell_acc[(sr.unit_id, sr.condition)].append(getattr(sr, metric))
    for (unit_id, cond), vals in sorted(cell_acc.items()):
        model.heatmap.append(
            HeatCell(unit_id=unit_id, condition=cond, metric_value=sum(vals) / len(vals))
        )

    return model


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
