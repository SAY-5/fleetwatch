"""Trend alerts over drift results.

An alert fires when a (unit, class) metric degrades beyond a threshold across at
least N windows. Each alert carries the magnitude of the drop and the condition
slice where the metric is worst, so an operator sees both "what degraded" and
"under which operating condition it is worst".
"""

from __future__ import annotations

from dataclasses import dataclass

from .drift import DriftResult


@dataclass
class Alert:
    unit_id: str
    cls: str
    metric: str
    drop_pp: float            # magnitude of the drop in percentage points
    n_windows: int
    worst_condition: str | None
    worst_condition_value: float | None
    message: str


def _format_pp(delta: float) -> float:
    return round(-delta * 100.0, 1)


def evaluate(
    drift: DriftResult,
    min_windows: int = 3,
    worst_condition: str | None = None,
    worst_condition_value: float | None = None,
) -> Alert | None:
    """Return an alert when the drift result clears the firing criteria."""
    if drift.n_windows < min_windows:
        return None
    if not drift.drifting:
        return None

    drop_pp = _format_pp(drift.delta)
    cond_suffix = ""
    if worst_condition is not None:
        cond_suffix = f"; worst under {worst_condition}"
    message = (
        f"{drift.unit_id}'s {drift.metric} on {drift.cls} dropped {drop_pp}pp "
        f"over the last {drift.n_windows} windows{cond_suffix}"
    )
    return Alert(
        unit_id=drift.unit_id,
        cls=drift.cls,
        metric=drift.metric,
        drop_pp=drop_pp,
        n_windows=drift.n_windows,
        worst_condition=worst_condition,
        worst_condition_value=worst_condition_value,
        message=message,
    )
