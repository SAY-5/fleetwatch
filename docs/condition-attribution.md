# Condition-sliced drift attribution

When a (unit, class) metric degrades, drift alone says *that* it fell, not *why*.
Attribution identifies which operating condition drove the fall, so an alert can
say "recall collapsed because of night, not because of rain or distance".

## How it works

The condition has three axes: `lighting`, `weather`, `distance`. For one
(unit, class), attribution reads the per-window condition slices from
`slice_metrics`, splits each slice key (`lighting|weather|distance`) back into
its axis values, and for every axis value computes:

- the mean metric over an early window segment (first `early_frac` of windows),
- the mean over a late segment (last `late_frac` of windows),
- the delta `late - early`.

The driver is the axis value with the most negative delta. Axis values that do
not appear in both segments are skipped, so a slice that is simply absent late in
the run is not mistaken for a degradation.

## Why per-axis, not per-full-slice

A unit that degrades only at night degrades across every
`night|*|*` slice. Ranking full slice keys would split that signal across nine
cells; ranking per axis value pools all night frames into a single
`lighting=night` figure, which then clearly outranks any single weather or
distance value.

## Validation

The test suite generates a unit whose recall drops only when `lighting=night`
(across all weather and distance values) and asserts that attribution names
`lighting=night` as the driver for the affected classes, with a delta roughly
double that of the next axis value. Weather and distance show milder deltas only
because night frames bleed into their averages; night still dominates.

## In the dashboard

The overview page lists each attributed degradation (unit, class, driver, delta)
and renders the condition heatmap coloured by the late-minus-early delta per
(unit x condition) cell: red cells degraded most, green improved.
