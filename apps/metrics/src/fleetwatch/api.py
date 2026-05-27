"""FastAPI dashboard. Fleshed out in the dashboard stage."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="FleetWatch")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
