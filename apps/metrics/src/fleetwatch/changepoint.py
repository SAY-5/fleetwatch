"""CUSUM change-point detection for metric drift.

The naive drift flag in ``drift.py`` fits a single slope, which cannot tell a
sudden cliff (a sensor failing at one window) from a slow ramp (a lens fogging
over many windows). This module adds a CUSUM detector that locates the window
where a (unit, class) metric structurally shifted, and a classifier that labels
the shift as a *cliff* or *drift*.

CUSUM (cumulative sum) tracks the running sum of deviations of the metric from a
reference mean. A persistent downward shift makes the negative cumulative sum
grow; the window of the maximum excursion is the most likely change point. The
classifier then compares the metric immediately around that window: a large drop
concentrated in one or two windows is a cliff, a comparable drop spread over many
windows is gradual drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ShiftKind(StrEnum):
    NONE = "none"
    CLIFF = "cliff"
    DRIFT = "drift"


@dataclass
class ChangePoint:
    detected: bool
    kind: ShiftKind
    window: int | None          # window index where the shift is located
    magnitude: float            # size of the level drop (pre-mean minus post-mean)
    pre_mean: float
    post_mean: float
    transition_windows: int     # how many windows the drop is spread over


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def detect(
    series: list[tuple[int, float]],
    threshold: float = 0.10,
    drift_slack: float = 0.005,
    cliff_transition: int = 2,
) -> ChangePoint:
    """Locate and classify a downward level shift in a metric series.

    ``threshold`` is the minimum pre-vs-post level drop (in metric points) to
    report a change point. ``drift_slack`` is the CUSUM allowance that ignores
    tiny fluctuations. ``cliff_transition`` is the maximum number of windows the
    drop may span to count as a cliff rather than gradual drift.
    """
    if len(series) < 4:
        return ChangePoint(False, ShiftKind.NONE, None, 0.0, 0.0, 0.0, 0)

    windows = [w for w, _ in series]
    values = [v for _, v in series]
    reference = _mean(values)

    # CUSUM of deviations from the global mean. For a downward level shift the
    # cumulative sum rises while values sit above the mean (before the shift)
    # and falls afterward, so it peaks at the change point. The drift_slack
    # allowance keeps tiny fluctuations from moving the estimate. The argmax is
    # taken over interior points so the split always leaves a non-empty side.
    cusum = 0.0
    peak = float("-inf")
    best_idx = 0
    for i, v in enumerate(values):
        cusum += v - reference - drift_slack
        if 0 <= i < len(values) - 1 and cusum > peak:
            peak = cusum
            best_idx = i

    # Split at the change point and compare the level before and after.
    pre = values[: best_idx + 1]
    post = values[best_idx + 1 :]
    pre_mean = _mean(pre)
    post_mean = _mean(post)
    magnitude = pre_mean - post_mean

    if magnitude < threshold:
        return ChangePoint(False, ShiftKind.NONE, None, magnitude, pre_mean, post_mean, 0)

    # Classify cliff vs drift by how abruptly the level moves at the change
    # point. Compare a short pre-window run mean to a short post-window run mean
    # right at the boundary: a cliff drops most of the magnitude across the
    # single boundary step, whereas drift spreads it out. To stay robust to the
    # per-window sampling noise of real fleet data, smooth with a small run and
    # measure the fraction of the total drop realised across the boundary.
    run = 3
    near_pre = values[max(0, best_idx - run + 1) : best_idx + 1]
    near_post = values[best_idx + 1 : best_idx + 1 + run]
    step_drop = _mean(near_pre) - _mean(near_post)
    boundary_fraction = step_drop / magnitude if magnitude > 0 else 0.0

    # The transition span is how many windows it takes to realise most of the
    # drop; reported for the dashboard. A cliff realises >=60% at the boundary.
    lo = post_mean + 0.1 * magnitude
    hi = pre_mean - 0.1 * magnitude
    transition = max(1, sum(1 for v in values if lo < v < hi))

    if boundary_fraction >= 0.6:
        kind = ShiftKind.CLIFF
        transition = min(transition, cliff_transition)
    else:
        kind = ShiftKind.DRIFT
    change_window = windows[min(best_idx + 1, len(windows) - 1)]
    return ChangePoint(
        detected=True,
        kind=kind,
        window=change_window,
        magnitude=magnitude,
        pre_mean=pre_mean,
        post_mean=post_mean,
        transition_windows=transition,
    )
