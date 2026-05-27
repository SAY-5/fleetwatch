"""Smoke tests for the fleet-scale benchmark and the regression gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fleetwatch import bench
from fleetwatch.aggregator import aggregator_path


def test_smoke_bench_runs() -> None:
    result = bench.run("smoke")
    assert result.n_detections > 0
    assert result.python_sec >= 0.0
    assert result.ingest_per_sec > 0.0
    # the binary protocol is materially smaller than JSON
    assert result.binary_bytes < result.json_bytes


def test_bench_without_aggregator_reports_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the binary is absent the cpp timings are NaN but the bench still runs.
    monkeypatch.setattr(bench, "aggregator_path", lambda: None)
    result = bench.run("smoke")
    assert math.isnan(result.cpp_sec)
    assert math.isnan(result.speedup)
    assert result.python_sec >= 0.0


@pytest.mark.skipif(aggregator_path() is None, reason="aggregator not built")
def test_smoke_bench_reports_speedup() -> None:
    result = bench.run("smoke")
    assert not math.isnan(result.cpp_sec)
    assert result.speedup > 0.0


def test_regress_writes_baseline_then_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(bench, "_BASELINE", baseline)

    # first run writes the baseline and passes
    assert bench.regress(0.30, "smoke") == 0
    assert baseline.exists()
    stored = json.loads(baseline.read_text())
    assert "speedup" in stored


def test_regress_passes_against_low_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A baseline of 0 means any non-negative speedup clears the floor, so the
    # compare-and-pass branch runs without needing the C++ binary.
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"scale": "smoke", "speedup": 0.0}))
    monkeypatch.setattr(bench, "_BASELINE", baseline)
    assert bench.regress(0.30, "smoke") == 0


@pytest.mark.skipif(aggregator_path() is None, reason="aggregator not built")
def test_regress_flags_a_collapsed_speedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    # an unrealistically high baseline forces the current run under the floor
    baseline.write_text(json.dumps({"scale": "smoke", "speedup": 1000.0}))
    monkeypatch.setattr(bench, "_BASELINE", baseline)
    assert bench.regress(0.30, "smoke") == 1


def test_main_runs_default_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["fleetwatch-bench", "--scale", "smoke"])
    assert bench.main() == 0


def test_main_regress_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"scale": "smoke", "speedup": 0.0}))
    monkeypatch.setattr(bench, "_BASELINE", baseline)
    monkeypatch.setattr(
        "sys.argv", ["fleetwatch-bench", "--scale", "smoke", "--regress", "0.30"]
    )
    assert bench.main() == 0
