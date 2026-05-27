"""Tests for the dashboard view-model builder."""

from __future__ import annotations

from fleetwatch.dashboard import build_model
from fleetwatch.pipeline import build_rows
from fleetwatch.sim import DegradeSpec, FleetConfig, generate


def _degrading_run():
    cfg = FleetConfig(
        units=3,
        frames=60,
        seed=7,
        base_recall=0.95,
        base_fp_rate=0.05,
        degrade=[DegradeSpec(unit_id="unit-0", per_window_drop=0.15)],
    )
    records = list(generate(cfg))
    return build_rows("demo", records, frames_per_window=10)


def test_model_has_units_and_heatmap() -> None:
    unit_rows, slice_rows = _degrading_run()
    model = build_model("demo", unit_rows, slice_rows)
    assert len(model.units) == 3
    assert model.conditions
    assert model.heatmap


def test_degrading_unit_produces_an_alert() -> None:
    unit_rows, slice_rows = _degrading_run()
    model = build_model("demo", unit_rows, slice_rows)
    assert any(a.unit_id == "unit-0" for a in model.alerts)


def test_drift_rows_cover_all_pairs() -> None:
    unit_rows, slice_rows = _degrading_run()
    model = build_model("demo", unit_rows, slice_rows)
    pairs_in_rows = {(r.unit_id, r.cls) for r in unit_rows}
    pairs_in_model = {(d.unit_id, d.cls) for d in model.drift_rows}
    assert pairs_in_rows == pairs_in_model
