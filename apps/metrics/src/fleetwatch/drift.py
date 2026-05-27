"""Drift computation over a per-(unit, class) metric series.

Given a time series of a metric (recall by default) across windows, drift is
summarised by the ordinary-least-squares slope plus a simple change-point flag:
the largest drop between the mean of an early segment and a late segment. The
CUSUM change-point detector that supersedes the naive flag arrives in a later
layer; this module provides the slope and the magnitude used by alerts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftResult:
    unit_id: str
    cls: str
    metric: str
    slope: float          # per-window change (negative means degrading)
    start_value: float
    end_value: float
    delta: float          # end - start of the fitted line
    drifting: bool        # delta below the (negative) threshold over the window span
    n_windows: int


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return num / den


def compute_drift(
    unit_id: str,
    cls: str,
    series: list[tuple[int, float]],
    metric: str = "recall",
    drop_threshold: float = 0.10,
) -> DriftResult:
    """Fit a line to the series and flag a degradation beyond ``drop_threshold``.

    ``drop_threshold`` is expressed in metric points (for example, 0.10 = 10pp).
    A series is flagged as drifting when the fitted line falls by at least the
    threshold from its first to its last window.
    """
    windows = [float(w) for w, _ in series]
    values = [float(v) for _, v in series]
    slope = _ols_slope(windows, values)

    if len(series) >= 2:
        span = windows[-1] - windows[0]
        intercept = (sum(values) / len(values)) - slope * (sum(windows) / len(windows))
        start_value = slope * windows[0] + intercept
        end_value = slope * windows[-1] + intercept
        delta = slope * span
    else:
        start_value = values[0] if values else 0.0
        end_value = start_value
        delta = 0.0

    drifting = delta <= -drop_threshold
    return DriftResult(
        unit_id=unit_id,
        cls=cls,
        metric=metric,
        slope=slope,
        start_value=start_value,
        end_value=end_value,
        delta=delta,
        drifting=drifting,
        n_windows=len(series),
    )
