"""Tests for condition-sliced drift attribution."""

from __future__ import annotations

from fleetwatch.attribution import attribute
from fleetwatch.pipeline import build_rows
from fleetwatch.schema import Lighting
from fleetwatch.sim import DegradeSpec, FleetConfig, generate


def _night_only_run(frames: int = 600):
    cfg = FleetConfig(
        units=1,
        frames=frames,
        seed=17,
        base_recall=0.95,
        base_fp_rate=0.0,
        frames_per_window=20,
        degrade=[
            DegradeSpec(unit_id="unit-0", per_window_drop=0.04, only_lighting=Lighting.NIGHT)
        ],
    )
    records = list(generate(cfg))
    return build_rows("attr", records, frames_per_window=20)


def test_night_only_degradation_is_attributed_to_lighting_night() -> None:
    _, slice_rows = _night_only_run()
    # attribute the worst class (whichever degrades) and assert lighting=night
    classes = sorted({r.cls for r in slice_rows if r.unit_id == "unit-0"})
    drivers = [attribute("unit-0", c, slice_rows, "recall") for c in classes]
    # at least one class is driven by the night-lighting degradation
    night_drivers = [
        a for a in drivers if a.driver_axis == "lighting" and a.driver_value == "night"
    ]
    assert night_drivers, [(a.driver_axis, a.driver_value, a.driver_delta) for a in drivers]
    # and the night driver should not be misattributed to weather or distance
    for a in night_drivers:
        # the ranked top is the night-lighting cell
        top = a.ranked[0]
        assert top.axis == "lighting"
        assert top.value == "night"


def test_night_driver_outranks_weather_and_distance() -> None:
    _, slice_rows = _night_only_run()
    classes = sorted({r.cls for r in slice_rows if r.unit_id == "unit-0"})
    found = False
    for c in classes:
        a = attribute("unit-0", c, slice_rows, "recall")
        if a.driver_axis != "lighting" or a.driver_value != "night":
            continue
        found = True
        # weather and distance axis deltas should be milder than the night drop
        night_delta = next(
            r.delta for r in a.ranked if r.axis == "lighting" and r.value == "night"
        )
        for r in a.ranked:
            if r.axis in ("weather", "distance"):
                assert r.delta > night_delta  # less negative than the night driver
    assert found


def test_stable_unit_has_no_driver() -> None:
    cfg = FleetConfig(units=1, frames=300, seed=3, base_recall=0.95, base_fp_rate=0.0)
    _, slice_rows = build_rows("attr", list(generate(cfg)), frames_per_window=20)
    a = attribute("unit-0", "car", slice_rows, "recall")
    # a stable unit either has no driver or only a tiny negative delta
    assert a.driver_axis is None or a.driver_delta > -0.1
