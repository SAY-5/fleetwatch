"""Tests for CUSUM change-point detection and cliff-vs-drift classification."""

from __future__ import annotations

from fleetwatch.changepoint import ShiftKind, detect


def _cliff(n: int, at: int, high: float, low: float) -> list[tuple[int, float]]:
    return [(w, high if w < at else low) for w in range(n)]


def _ramp(n: int, start: float, end: float) -> list[tuple[int, float]]:
    step = (end - start) / (n - 1)
    return [(w, start + step * w) for w in range(n)]


def test_cliff_is_detected_at_the_exact_window() -> None:
    series = _cliff(24, at=12, high=0.92, low=0.45)
    cp = detect(series)
    assert cp.detected
    assert cp.kind is ShiftKind.CLIFF
    assert cp.window == 12
    assert cp.magnitude > 0.4


def test_cliff_at_a_different_window() -> None:
    series = _cliff(20, at=7, high=0.9, low=0.5)
    cp = detect(series)
    assert cp.detected
    assert cp.kind is ShiftKind.CLIFF
    assert cp.window == 7


def test_gradual_ramp_is_classified_as_drift_not_cliff() -> None:
    series = _ramp(24, start=0.95, end=0.45)
    cp = detect(series)
    assert cp.detected
    assert cp.kind is ShiftKind.DRIFT
    assert cp.transition_windows > 2


def test_flat_series_has_no_change_point() -> None:
    series = [(w, 0.9) for w in range(20)]
    cp = detect(series)
    assert not cp.detected
    assert cp.kind is ShiftKind.NONE


def test_small_dip_below_threshold_not_flagged() -> None:
    series = _cliff(20, at=10, high=0.90, low=0.86)
    cp = detect(series, threshold=0.10)
    assert not cp.detected


def test_short_series_returns_no_change_point() -> None:
    assert not detect([(0, 0.9), (1, 0.5)]).detected


def test_sim_cliff_classified_as_cliff_at_injected_window() -> None:
    from fleetwatch.dashboard import build_model
    from fleetwatch.pipeline import build_rows
    from fleetwatch.sim import DegradeSpec, FleetConfig, generate

    cfg = FleetConfig(
        units=1,
        frames=400,
        seed=5,
        base_recall=0.97,
        base_fp_rate=0.0,
        frames_per_window=20,
        degrade=[DegradeSpec(unit_id="unit-0", cliff_window=6, cliff_drop=0.7)],
    )
    ur, sr = build_rows("cp", list(generate(cfg)), frames_per_window=20)
    model = build_model("cp", ur, sr)
    cliffs = [d for d in model.drift_rows if d.change_kind == "cliff"]
    assert cliffs
    assert all(d.change_window == 6 for d in cliffs)


def test_sim_gradual_ramp_classified_as_drift() -> None:
    from fleetwatch.dashboard import build_model
    from fleetwatch.pipeline import build_rows
    from fleetwatch.sim import DegradeSpec, FleetConfig, generate

    cfg = FleetConfig(
        units=1,
        frames=400,
        seed=5,
        base_recall=0.97,
        base_fp_rate=0.0,
        frames_per_window=20,
        degrade=[DegradeSpec(unit_id="unit-0", per_window_drop=0.05)],
    )
    ur, sr = build_rows("cp", list(generate(cfg)), frames_per_window=20)
    model = build_model("cp", ur, sr)
    detected = [d for d in model.drift_rows if d.change_kind != "none"]
    assert detected
    assert all(d.change_kind == "drift" for d in detected)
