"""PostgreSQL metric store.

Persists per-window, per-(unit, class) metrics and per-(unit, class, condition)
slices. The schema is intentionally narrow: one row per metric observation so
drift and condition slicing are plain SQL aggregations. Connection is via a DSN
(``FLEETWATCH_DSN``); in CI a testcontainers Postgres provides it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS unit_metrics (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT    NOT NULL,
    unit_id     TEXT    NOT NULL,
    cls         TEXT    NOT NULL,
    window_idx  INTEGER NOT NULL,
    tp          INTEGER NOT NULL,
    fp          INTEGER NOT NULL,
    fn          INTEGER NOT NULL,
    precision   DOUBLE PRECISION NOT NULL,
    recall      DOUBLE PRECISION NOT NULL,
    f1          DOUBLE PRECISION NOT NULL,
    ap          DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_unit_metrics_lookup
    ON unit_metrics (run_id, unit_id, cls, window_idx);

CREATE TABLE IF NOT EXISTS slice_metrics (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT    NOT NULL,
    unit_id     TEXT    NOT NULL,
    cls         TEXT    NOT NULL,
    window_idx  INTEGER NOT NULL,
    condition   TEXT    NOT NULL,
    precision   DOUBLE PRECISION NOT NULL,
    recall      DOUBLE PRECISION NOT NULL,
    ap          DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_slice_metrics_lookup
    ON slice_metrics (run_id, unit_id, cls, condition, window_idx);
"""


@dataclass
class UnitMetricRow:
    run_id: str
    unit_id: str
    cls: str
    window_idx: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap: float


@dataclass
class SliceMetricRow:
    run_id: str
    unit_id: str
    cls: str
    window_idx: int
    condition: str
    precision: float
    recall: float
    ap: float


def dsn() -> str:
    value = os.environ.get("FLEETWATCH_DSN")
    if not value:
        raise RuntimeError("FLEETWATCH_DSN is not set")
    return value


@contextmanager
def connect(dsn_str: str | None = None) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(dsn_str or dsn())
    try:
        yield conn
    finally:
        conn.close()


def init_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def insert_unit_metrics(conn: psycopg.Connection, rows: list[UnitMetricRow]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO unit_metrics
               (run_id, unit_id, cls, window_idx, tp, fp, fn, precision, recall, f1, ap)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [
                (
                    r.run_id, r.unit_id, r.cls, r.window_idx, r.tp, r.fp, r.fn,
                    r.precision, r.recall, r.f1, r.ap,
                )
                for r in rows
            ],
        )
    conn.commit()


def insert_slice_metrics(conn: psycopg.Connection, rows: list[SliceMetricRow]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO slice_metrics
               (run_id, unit_id, cls, window_idx, condition, precision, recall, ap)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            [
                (
                    r.run_id, r.unit_id, r.cls, r.window_idx, r.condition,
                    r.precision, r.recall, r.ap,
                )
                for r in rows
            ],
        )
    conn.commit()


def series(
    conn: psycopg.Connection, run_id: str, unit_id: str, cls: str, metric: str = "recall"
) -> list[tuple[int, float]]:
    """Return the (window_idx, metric) series for one unit and class, ordered."""
    if metric not in {"precision", "recall", "f1", "ap"}:
        raise ValueError(f"unknown metric {metric}")
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT window_idx, {metric} FROM unit_metrics
                WHERE run_id=%s AND unit_id=%s AND cls=%s
                ORDER BY window_idx""",
            (run_id, unit_id, cls),
        )
        return [(int(w), float(v)) for w, v in cur.fetchall()]


def units_and_classes(conn: psycopg.Connection, run_id: str) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT unit_id, cls FROM unit_metrics WHERE run_id=%s ORDER BY unit_id, cls",
            (run_id,),
        )
        return [(str(u), str(c)) for u, c in cur.fetchall()]
