"""Condition-sliced drift attribution.

When a (unit, class) metric degrades, attribution answers *which operating
condition drove it*. The condition has three axes (lighting, weather,
distance_band); attribution measures, per axis value, how much the metric fell
from an early window segment to a late segment, and reports the axis value with
the largest fall.

The slice series come from ``slice_metrics`` rows, whose condition key is
``lighting|weather|distance``. We split that key back into axes so a degradation
isolated to, say, ``lighting=night`` is attributed to the lighting axis rather
than to a co-occurring weather or distance value.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .store import SliceMetricRow

AXES = ("lighting", "weather", "distance")


@dataclass
class AxisDelta:
    axis: str
    value: str
    early_mean: float
    late_mean: float
    delta: float  # late - early (negative means degradation)
    n_early: int
    n_late: int


@dataclass
class Attribution:
    unit_id: str
    cls: str
    metric: str
    driver_axis: str | None
    driver_value: str | None
    driver_delta: float
    ranked: list[AxisDelta]


def _split(condition_key: str) -> dict[str, str]:
    parts = condition_key.split("|")
    return dict(zip(AXES, parts, strict=True))


def attribute(
    unit_id: str,
    cls: str,
    slice_rows: list[SliceMetricRow],
    metric: str = "recall",
    early_frac: float = 0.4,
    late_frac: float = 0.4,
) -> Attribution:
    """Attribute a (unit, class) degradation to the worst condition axis value.

    For each axis value the metric is averaged over an early window segment and a
    late window segment; the delta is ``late - early``. The driver is the axis
    value with the most negative delta. Values seen in only one segment are
    skipped so a missing slice does not masquerade as a degradation.
    """
    rows = [r for r in slice_rows if r.unit_id == unit_id and r.cls == cls]
    windows = sorted({r.window_idx for r in rows})

    ranked: list[AxisDelta] = []
    if len(windows) >= 2:
        n = len(windows)
        early_cut = windows[max(0, int(n * early_frac) - 1)]
        late_cut = windows[min(n - 1, n - int(n * late_frac))]

        # Accumulate metric values per (axis, value) for the early/late segments.
        early: dict[tuple[str, str], list[float]] = defaultdict(list)
        late: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in rows:
            axes = _split(r.condition)
            value = getattr(r, metric)
            for axis in AXES:
                key = (axis, axes[axis])
                if r.window_idx <= early_cut:
                    early[key].append(value)
                if r.window_idx >= late_cut:
                    late[key].append(value)

        for key in sorted(set(early) & set(late)):
            axis, value = key
            e = sum(early[key]) / len(early[key])
            la = sum(late[key]) / len(late[key])
            ranked.append(
                AxisDelta(
                    axis=axis,
                    value=value,
                    early_mean=e,
                    late_mean=la,
                    delta=la - e,
                    n_early=len(early[key]),
                    n_late=len(late[key]),
                )
            )
        ranked.sort(key=lambda d: d.delta)

    if ranked and ranked[0].delta < 0:
        top = ranked[0]
        return Attribution(
            unit_id=unit_id,
            cls=cls,
            metric=metric,
            driver_axis=top.axis,
            driver_value=top.value,
            driver_delta=top.delta,
            ranked=ranked,
        )
    return Attribution(
        unit_id=unit_id,
        cls=cls,
        metric=metric,
        driver_axis=None,
        driver_value=None,
        driver_delta=0.0,
        ranked=ranked,
    )
