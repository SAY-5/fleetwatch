"""HTTP-level tests for the dashboard using the demo JSONL data source."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fleetwatch.api import app
from fleetwatch.sim import DegradeSpec, FleetConfig, generate


@pytest.fixture()
def demo_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = FleetConfig(
        units=3,
        frames=60,
        seed=7,
        base_recall=0.95,
        base_fp_rate=0.05,
        degrade=[DegradeSpec(unit_id="unit-0", per_window_drop=0.15)],
    )
    out = tmp_path / "demo.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for rec in generate(cfg):
            fh.write(rec.model_dump_json(by_alias=True))
            fh.write("\n")
    monkeypatch.setenv("FLEETWATCH_DEMO_RUN", str(out))
    return out


def test_healthz() -> None:
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}


def test_overview_renders(demo_run: Path) -> None:
    client = TestClient(app)
    resp = client.get("/?run_id=demo")
    assert resp.status_code == 200
    assert "Fleet overview" in resp.text
    assert "Condition heatmap" in resp.text


def test_unit_drilldown_renders(demo_run: Path) -> None:
    client = TestClient(app)
    resp = client.get("/unit/unit-0?run_id=demo")
    assert resp.status_code == 200
    assert "Per-class drift" in resp.text


def test_alerts_api_flags_degrading_unit(demo_run: Path) -> None:
    client = TestClient(app)
    alerts = client.get("/api/alerts?run_id=demo").json()
    assert any(a["unit_id"] == "unit-0" for a in alerts)


def test_unknown_unit_is_404(demo_run: Path) -> None:
    client = TestClient(app)
    assert client.get("/unit/unit-99?run_id=demo").status_code == 404
