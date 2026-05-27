"""Tests for trend-alert firing."""

from __future__ import annotations

from fleetwatch.alerts import evaluate
from fleetwatch.drift import compute_drift


def _drift(values: list[float]):
    series = list(enumerate(values))
    return compute_drift("unit-3", "pedestrian", series, drop_threshold=0.10)


def test_alert_fires_on_sustained_drop() -> None:
    drift = _drift([0.9, 0.85, 0.8, 0.72, 0.6])
    alert = evaluate(drift, worst_condition="night|rain|far")
    assert alert is not None
    assert alert.unit_id == "unit-3"
    assert alert.drop_pp > 0
    assert "pedestrian" in alert.message
    assert "night|rain|far" in alert.message


def test_no_alert_when_stable() -> None:
    drift = _drift([0.9, 0.9, 0.89, 0.9, 0.91])
    assert evaluate(drift) is None


def test_no_alert_below_min_windows() -> None:
    drift = _drift([0.9, 0.4])
    assert evaluate(drift, min_windows=3) is None


def test_drop_magnitude_in_pp() -> None:
    drift = _drift([1.0, 0.9, 0.8, 0.7, 0.6])
    alert = evaluate(drift)
    assert alert is not None
    # fitted delta -0.4 -> 40.0 pp
    assert alert.drop_pp == 40.0
