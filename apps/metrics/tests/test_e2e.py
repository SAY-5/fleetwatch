"""End-to-end: generate -> ingest -> compute -> store -> dashboard -> alert.

A fleet with one deliberately degrading unit flows through the full pipeline.
The test asserts golden metric values for a perfect baseline window and that the
degrading unit both shows up in the dashboard model and fires a trend alert.
The store leg runs against a real Postgres via testcontainers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleetwatch import metrics_ref, store
from fleetwatch.aggregator import aggregator_path, compute_cpp
from fleetwatch.dashboard import build_model
from fleetwatch.ingest import read_jsonl
from fleetwatch.pipeline import build_rows
from fleetwatch.sim import DegradeSpec, FleetConfig, generate

pytest.importorskip("testcontainers.postgres")
from testcontainers.postgres import PostgresContainer  # noqa: E402

pytestmark = pytest.mark.integration

DEGRADING_UNIT = "unit-0"


def _run_config() -> FleetConfig:
    return FleetConfig(
        units=4,
        frames=80,
        seed=21,
        base_recall=0.97,
        base_fp_rate=0.04,
        frames_per_window=10,
        degrade=[DegradeSpec(unit_id=DEGRADING_UNIT, per_window_drop=0.13)],
    )


def test_perfect_unit_golden_metrics() -> None:
    # A unit with full recall and no false positives has perfect golden metrics.
    cfg = FleetConfig(units=1, frames=40, seed=21, base_recall=1.0, base_fp_rate=0.0)
    records = list(generate(cfg))
    m = metrics_ref.compute(records, 0.5)
    assert m.map == pytest.approx(1.0, abs=1e-12)
    assert m.micro_precision == pytest.approx(1.0, abs=1e-12)
    assert m.micro_recall == pytest.approx(1.0, abs=1e-12)
    assert m.micro_f1 == pytest.approx(1.0, abs=1e-12)
    for c in m.per_class:
        assert c.fp == 0
        assert c.fn == 0


@pytest.mark.skipif(aggregator_path() is None, reason="aggregator not built")
def test_cpp_and_python_agree_on_full_run() -> None:
    records = list(generate(_run_config()))
    ref = metrics_ref.compute(records, 0.5)
    cpp = compute_cpp(records, 0.5)
    assert cpp.map == pytest.approx(ref.map, abs=1e-9)
    assert cpp.micro_recall == pytest.approx(ref.micro_recall, abs=1e-9)


def test_full_pipeline_through_store_flags_degrading_unit(tmp_path: Path) -> None:
    cfg = _run_config()

    # generate -> jsonl
    run_file = tmp_path / "run.jsonl"
    with run_file.open("w", encoding="utf-8") as fh:
        for rec in generate(cfg):
            fh.write(rec.model_dump_json(by_alias=True))
            fh.write("\n")

    # ingest
    records = list(read_jsonl(run_file))
    assert len(records) == cfg.units * cfg.frames

    # compute + window
    unit_rows, slice_rows = build_rows("e2e", records, frames_per_window=10)

    # store
    with PostgresContainer("postgres:16-alpine") as pg:
        dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://").replace(
            "postgresql+psycopg://", "postgresql://"
        )
        with store.connect(dsn) as conn:
            store.init_schema(conn)
            store.insert_unit_metrics(conn, unit_rows)
            store.insert_slice_metrics(conn, slice_rows)

            # the degrading unit's recall series must decline across windows
            pairs = store.units_and_classes(conn, "e2e")
            unit0_classes = [c for u, c in pairs if u == DEGRADING_UNIT]
            assert unit0_classes
            declining = False
            for cls in unit0_classes:
                s = store.series(conn, "e2e", DEGRADING_UNIT, cls, "recall")
                if len(s) >= 2 and s[-1][1] < s[0][1] - 0.1:
                    declining = True
            assert declining

    # dashboard model: degrading unit appears and fires an alert
    model = build_model("e2e", unit_rows, slice_rows)
    assert any(u.unit_id == DEGRADING_UNIT for u in model.units)
    unit0_alerts = [a for a in model.alerts if a.unit_id == DEGRADING_UNIT]
    assert unit0_alerts
    assert any(a.drop_pp > 0 for a in unit0_alerts)
