# Change-point detection (CUSUM)

The naive drift flag fits a single slope to a metric series. That cannot tell a
sudden cliff (a sensor failing at one window) from a slow ramp (a lens fogging
over many windows): both can produce the same average slope. FleetWatch adds a
CUSUM change-point detector that locates *where* a (unit, class) metric shifted
and a classifier that labels the shift as a cliff or as gradual drift.

## CUSUM

For a metric series, let `mean` be the global mean. The detector accumulates the
cumulative sum of deviations `value - mean - drift_slack`. While values sit above
the mean (before a downward shift) the cumulative sum rises; after the shift it
falls. The sum therefore peaks at the change point, so the detector reports the
window just after the argmax of the cumulative sum.

`drift_slack` (default `0.005`) is a small allowance that stops ordinary
window-to-window noise from moving the estimate. A change point is only reported
when the pre-change mean exceeds the post-change mean by at least `threshold`
(default `0.10`, i.e. 10 percentage points).

## Cliff vs drift

Once the change point is located, the classifier measures how abruptly the level
moves there. It compares a short run of windows just before the boundary to a
short run just after and computes the fraction of the total drop realised across
that single boundary step:

- **cliff**: at least 60% of the drop happens at the boundary; the metric falls
  in essentially one window (a sensor failure, an occlusion).
- **drift**: the drop is spread across many windows (a lens fogging, slow
  calibration loss).

Smoothing with a short run keeps the classifier robust to the per-window
sampling noise of real fleet data, where individual windows scatter even when
the underlying level is flat.

## Parameters

| param             | default | meaning                                            |
|-------------------|---------|----------------------------------------------------|
| `threshold`       | 0.10    | minimum pre-vs-post level drop to report           |
| `drift_slack`     | 0.005   | CUSUM allowance that ignores tiny fluctuations     |
| `cliff_transition`| 2       | max windows a cliff may span in the reported count |

## Validation

The test suite injects a cliff at a known window and asserts the detector flags
that exact window and classifies it `cliff`; it injects a gradual ramp and
asserts the detector flags `drift`, not `cliff`. The same holds end to end on
synthetic fleet runs: a unit with `cliff_window=6` is reported as a cliff at
window 6, while a unit with a per-window ramp is reported as drift.
