"""FastAPI dashboard: fleet overview, per-unit drilldown, condition heatmap.

Data is read from the Postgres metric store when ``FLEETWATCH_DSN`` is set. For
a zero-dependency demo (and the end-to-end test) the app can also load a run
directly from a JSONL file named by ``FLEETWATCH_DEMO_RUN`` and compute the
metrics in process.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .dashboard import DashboardModel, build_model
from .ingest import read_jsonl
from .pipeline import build_rows
from .store import SliceMetricRow, UnitMetricRow, connect

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="FleetWatch")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _load_rows(run_id: str) -> tuple[list[UnitMetricRow], list[SliceMetricRow]]:
    """Load metric rows for a run from the demo JSONL or the Postgres store."""
    demo = os.environ.get("FLEETWATCH_DEMO_RUN")
    if demo:
        records = list(read_jsonl(Path(demo)))
        return build_rows(run_id, records, frames_per_window=10)

    if not os.environ.get("FLEETWATCH_DSN"):
        raise HTTPException(status_code=503, detail="no data source configured")

    unit_rows: list[UnitMetricRow] = []
    slice_rows: list[SliceMetricRow] = []
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT run_id, unit_id, cls, window_idx, tp, fp, fn,
                      precision, recall, f1, ap
               FROM unit_metrics WHERE run_id=%s ORDER BY unit_id, cls, window_idx""",
            (run_id,),
        )
        unit_rows = [UnitMetricRow(*row) for row in cur.fetchall()]
        cur.execute(
            """SELECT run_id, unit_id, cls, window_idx, condition, precision, recall, ap
               FROM slice_metrics WHERE run_id=%s
               ORDER BY unit_id, cls, condition, window_idx""",
            (run_id,),
        )
        slice_rows = [SliceMetricRow(*row) for row in cur.fetchall()]
    return unit_rows, slice_rows


def load_model(run_id: str) -> DashboardModel:
    unit_rows, slice_rows = _load_rows(run_id)
    if not unit_rows:
        raise HTTPException(status_code=404, detail=f"no metrics for run {run_id}")
    return build_model(run_id, unit_rows, slice_rows)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, run_id: str = "demo") -> HTMLResponse:
    model = load_model(run_id)
    return _TEMPLATES.TemplateResponse(
        request, "overview.html", {"model": model, "run_id": run_id}
    )


@app.get("/unit/{unit_id}", response_class=HTMLResponse)
def unit(request: Request, unit_id: str, run_id: str = "demo") -> HTMLResponse:
    model = load_model(run_id)
    drift_rows = [d for d in model.drift_rows if d.unit_id == unit_id]
    if not drift_rows:
        raise HTTPException(status_code=404, detail=f"no metrics for unit {unit_id}")
    alerts = [a for a in model.alerts if a.unit_id == unit_id]
    return _TEMPLATES.TemplateResponse(
        request,
        "unit.html",
        {"unit_id": unit_id, "drift_rows": drift_rows, "alerts": alerts, "run_id": run_id},
    )


@app.get("/api/alerts")
def api_alerts(run_id: str = "demo") -> list[dict[str, object]]:
    model = load_model(run_id)
    return [
        {
            "unit_id": a.unit_id,
            "class": a.cls,
            "metric": a.metric,
            "drop_pp": a.drop_pp,
            "n_windows": a.n_windows,
            "worst_condition": a.worst_condition,
            "message": a.message,
        }
        for a in model.alerts
    ]
