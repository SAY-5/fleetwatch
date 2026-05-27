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


def test_binary_request_has_magic_and_is_smaller() -> None:
    recs = [
        DetectionRecord.model_validate(
            {
                "unit_id": "u",
                "frame_id": 0,
                "timestamp": 0.0,
                "condition": {"lighting": "day", "weather": "clear", "distance_band": "near"},
                "detections": [{"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9}],
                "ground_truth": [{"class": "car", "bbox": [0, 0, 10, 10]}],
            }
        )
    ]
    binary = aggregator.build_binary_request(recs, 0.5)
    assert binary.startswith(b"FWB1")
    assert len(binary) < len(aggregator.build_request(recs, 0.5).encode())


def test_compute_cpp_rejects_unknown_wire() -> None:
    with pytest.raises(ValueError, match="unknown wire"):
        aggregator.compute_cpp([_record()], 0.5, binary_path="/bin/true", wire="protobuf")


def _fake_aggregator(tmp_path: Path, body: str, rc: int = 0) -> str:
    script = tmp_path / "fake-agg"
    script.write_text(f'#!/bin/sh\ncat > /dev/null\nprintf %s {body!r}\nexit {rc}\n')
    script.chmod(0o755)
    return str(script)


def test_compute_cpp_parses_a_canned_response(tmp_path: Path) -> None:
    response = (
        '{"iou_threshold":0.5,"map":1.0,"micro_precision":1.0,"micro_recall":1.0,'
        '"micro_f1":1.0,"per_class":[{"class":"car","tp":1,"fp":0,"fn":0,'
        '"precision":1.0,"recall":1.0,"f1":1.0,"ap":1.0}]}'
    )
    fake = _fake_aggregator(tmp_path, response)
    for wire in ("binary", "json"):
        m = aggregator.compute_cpp([_record()], 0.5, binary_path=fake, wire=wire)
        assert m.map == 1.0
        assert m.per_class[0].cls == "car"


def test_compute_cpp_raises_on_nonzero_exit(tmp_path: Path) -> None:
    fake = _fake_aggregator(tmp_path, "boom", rc=2)
    with pytest.raises(RuntimeError, match="aggregator failed"):
        aggregator.compute_cpp([_record()], 0.5, binary_path=fake)


def test_compute_cpp_missing_binary_raises() -> None:
    with pytest.raises(FileNotFoundError):
        aggregator.compute_cpp([_record()], 0.5, binary_path="/nonexistent/aggregator-binary")


def test_aggregator_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "fleetwatch-aggregator"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("FLEETWATCH_AGGREGATOR", str(fake))
    assert aggregator.aggregator_path() == str(fake)
