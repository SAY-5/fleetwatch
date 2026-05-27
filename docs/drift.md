# Drift

Drift tracks how a per-(unit, class) metric moves across time windows. A window
is a fixed number of consecutive frames (default 10); each window yields one
metric observation per (unit, class), stored in `unit_metrics`.

## Model

For a metric series `[(window, value), ...]` (recall by default), drift is
summarised by:

- **slope**: the ordinary-least-squares slope of value against window index. A
  negative slope means the metric is degrading.
- **delta**: the change of the fitted line from the first to the last window,
  `slope * (last_window - first_window)`. This is the magnitude used by alerts.
- **drifting**: `True` when `delta <= -drop_threshold` (default `0.10`, i.e. a
  10-percentage-point fall across the observed span).

A single-window series has slope `0` and is never flagged. The change-point
detector that distinguishes a sudden cliff from gradual drift is documented in
[changepoint.md](changepoint.md).

## Alerts

A trend alert fires for a (unit, class) when the series is drifting and spans at
least `min_windows` windows (default 3). The alert reports the drop in
percentage points and the condition slice where the metric is worst, for
example:

> unit-0's recall on pedestrian dropped 28.0pp over the last 8 windows; worst
> under night|rain|far
