"""Tests for drift slope and the degradation flag."""

from __future__ import annotations

import pytest

from fleetwatch.drift import compute_drift


def test_flat_series_has_zero_slope_and_no_drift() -> None:
    series = [(i, 0.9) for i in range(6)]
    r = compute_drift("unit-0", "car", series)
    assert r.slope == pytest.approx(0.0, abs=1e-12)
    assert not r.drifting


def test_declining_series_is_flagged() -> None:
    # recall drops 0.05 per window over 6 windows -> delta = -0.25.
    series = [(i, 0.9 - 0.05 * i) for i in range(6)]
    r = compute_drift("unit-0", "pedestrian", series, drop_threshold=0.10)
    assert r.slope < 0
    assert r.delta == pytest.approx(-0.25, abs=1e-9)
    assert r.drifting


def test_small_decline_below_threshold_not_flagged() -> None:
    series = [(i, 0.9 - 0.005 * i) for i in range(6)]
    r = compute_drift("unit-0", "car", series, drop_threshold=0.10)
    assert not r.drifting


def test_single_window_is_not_drift() -> None:
    r = compute_drift("unit-0", "car", [(0, 0.5)])
    assert r.slope == 0.0
    assert not r.drifting
    assert r.n_windows == 1


def test_improving_series_not_flagged() -> None:
    series = [(i, 0.5 + 0.05 * i) for i in range(6)]
    r = compute_drift("unit-0", "car", series)
    assert r.slope > 0
    assert not r.drifting
