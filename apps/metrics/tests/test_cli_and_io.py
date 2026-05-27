"""Tests for the sim CLI, ingest CLI and aggregator client error paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from fleetwatch import aggregator, ingest, sim_cli
from fleetwatch.schema import Condition, DetectionRecord, DistanceBand, Lighting, Weather


def _record(unit: str = "u", frame: int = 0) -> DetectionRecord:
    return DetectionRecord(
        unit_id=unit,
        frame_id=frame,
        timestamp=float(frame),
        condition=Condition(
            lighting=Lighting.DAY, weather=Weather.CLEAR, distance_band=DistanceBand.NEAR
        ),
        detections=[],
        ground_truth=[],
    )


def test_sim_cli_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "run.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["fleetwatch-sim", "--units", "2", "--frames", "5", "--seed", "1", "--out", str(out)],
    )
    sim_cli.main()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2 * 5


def test_ingest_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    out.write_text(_record().model_dump_json(by_alias=True) + "\n\n")
    records = list(ingest.read_jsonl(out))
    assert len(records) == 1
    assert records[0].unit_id == "u"


def test_ingest_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    out = tmp_path / "r.jsonl"
    out.write_text(_record().model_dump_json(by_alias=True) + "\n")
    monkeypatch.setattr("sys.argv", ["fleetwatch-ingest", str(out)])
    ingest.cli()
    assert "validated 1 records" in capsys.readouterr().out


def test_ingest_load_skips_blank_lines() -> None:
    line = _record().model_dump_json(by_alias=True)
    assert len(ingest.load([line, "", "   "])) == 1


def test_build_request_shape() -> None:
    req = aggregator.build_request([_record()], 0.5)
    assert '"iou_threshold": 0.5' in req
    assert '"frames"' in req


def test_compute_cpp_missing_binary_raises() -> None:
    with pytest.raises(FileNotFoundError):
        aggregator.compute_cpp([_record()], 0.5, binary_path="/nonexistent/aggregator-binary")


def test_aggregator_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "fleetwatch-aggregator"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("FLEETWATCH_AGGREGATOR", str(fake))
    assert aggregator.aggregator_path() == str(fake)
