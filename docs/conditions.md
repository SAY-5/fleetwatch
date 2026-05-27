# Condition slicing

Every detection record carries an operating condition with three axes:

- **lighting**: `day`, `night`, `dusk`
- **weather**: `clear`, `rain`, `fog`
- **distance_band**: `near`, `mid`, `far`

The condition is serialised to a stable slice key `lighting|weather|distance`
(for example `night|rain|far`) and stored per observation in `slice_metrics`,
so any metric can be sliced by condition with plain SQL.

## Why slice

Fleet-wide averages hide where a model actually fails. A unit can look healthy
on average while its recall collapses at night or in the rain. The dashboard's
condition heatmap shows the mean metric per (unit x condition) cell so the worst
operating conditions are visible at a glance: greener cells are higher recall,
redder cells degraded most.

## In the dashboard

The overview page renders the heatmap directly. Per-unit alerts name the worst
condition for the affected class, computed as the condition slice with the
lowest mean metric for that (unit, class). Condition-driven drift attribution,
which identifies which condition *drove* a degradation, is documented in
[condition-attribution.md](condition-attribution.md).
