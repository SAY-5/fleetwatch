"""Parity test: the C++ aggregator agrees with the Python reference."""

from __future__ import annotations

import pytest

from fleetwatch import metrics_ref
from fleetwatch.aggregator import aggregator_path, compute_cpp
from fleetwatch.sim import FleetConfig, generate

TOL = 1e-9

pytestmark = pytest.mark.skipif(
    aggregator_path() is None, reason="aggregator binary not built"
)


def _assert_metrics_close(a: metrics_ref.BatchMetrics, b: metrics_ref.BatchMetrics) -> None:
    assert a.map == pytest.approx(b.map, abs=TOL)
    assert a.micro_precision == pytest.approx(b.micro_precision, abs=TOL)
    assert a.micro_recall == pytest.approx(b.micro_recall, abs=TOL)
    assert a.micro_f1 == pytest.approx(b.micro_f1, abs=TOL)
    assert len(a.per_class) == len(b.per_class)
    for ca, cb in zip(a.per_class, b.per_class, strict=True):
        assert ca.cls == cb.cls
        assert ca.tp == cb.tp
        assert ca.fp == cb.fp
        assert ca.fn == cb.fn
        assert ca.precision == pytest.approx(cb.precision, abs=TOL)
        assert ca.recall == pytest.approx(cb.recall, abs=TOL)
        assert ca.f1 == pytest.approx(cb.f1, abs=TOL)
        assert ca.ap == pytest.approx(cb.ap, abs=TOL)


@pytest.mark.parametrize("seed", [1, 7, 42, 100])
def test_cpp_matches_python_reference(seed: int) -> None:
    records = list(generate(FleetConfig(units=5, frames=40, seed=seed)))
    ref = metrics_ref.compute(records, 0.5)
    cpp = compute_cpp(records, 0.5)
    _assert_metrics_close(ref, cpp)


def test_parity_on_empty_batch() -> None:
    ref = metrics_ref.compute([], 0.5)
    cpp = compute_cpp([], 0.5)
    _assert_metrics_close(ref, cpp)
