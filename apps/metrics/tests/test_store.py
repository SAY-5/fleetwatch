"""Integration test for the Postgres metric store via testcontainers."""

from __future__ import annotations

import pytest

from fleetwatch import store
from fleetwatch.pipeline import build_rows
from fleetwatch.sim import DegradeSpec, FleetConfig, generate

pytest.importorskip("testcontainers.postgres")
from testcontainers.postgres import PostgresContainer  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_dsn():
    with PostgresContainer("postgres:16-alpine") as pg:
        url = pg.get_connection_url()
        # testcontainers returns a SQLAlchemy-style URL; psycopg wants plain.
        dsn = url.replace("postgresql+psycopg2://", "postgresql://").replace(
            "postgresql+psycopg://", "postgresql://"
        )
        yield dsn


def test_store_roundtrip_and_series(pg_dsn: str) -> None:
    cfg = FleetConfig(
        units=2,
        frames=60,
        seed=7,
        degrade=[DegradeSpec(unit_id="unit-0", per_window_drop=0.12)],
    )
    records = list(generate(cfg))
    unit_rows, slice_rows = build_rows("run-test", records, frames_per_window=10)

    with store.connect(pg_dsn) as conn:
        store.init_schema(conn)
        store.insert_unit_metrics(conn, unit_rows)
        store.insert_slice_metrics(conn, slice_rows)

        pairs = store.units_and_classes(conn, "run-test")
        assert pairs
        assert any(u == "unit-0" for u, _ in pairs)

        # the degrading unit should have a recall series across windows
        cls = next(c for u, c in pairs if u == "unit-0")
        s = store.series(conn, "run-test", "unit-0", cls, "recall")
        assert len(s) >= 2
