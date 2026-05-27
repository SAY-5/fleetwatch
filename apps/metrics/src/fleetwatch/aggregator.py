"""Python client for the C++ aggregator subprocess.

Serialises a batch of detection records into the request JSON, invokes the
``fleetwatch-aggregator`` binary, and parses the response back into the same
``BatchMetrics`` dataclass the Python reference produces. If the binary is not
available the caller can fall back to ``metrics_ref.compute``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .metrics_ref import BatchMetrics, ClassMetric
from .schema import DetectionRecord

_ENV_VAR = "FLEETWATCH_AGGREGATOR"


def aggregator_path() -> str | None:
    """Resolve the aggregator binary from the env var, PATH or the build dir."""
    env = os.environ.get(_ENV_VAR)
    if env and Path(env).exists():
        return env
    found = shutil.which("fleetwatch-aggregator")
    if found:
        return found
    candidate = (
        Path(__file__).resolve().parents[4]
        / "apps"
        / "aggregator"
        / "build"
        / "fleetwatch-aggregator"
    )
    if candidate.exists():
        return str(candidate)
    return None


def build_request(records: list[DetectionRecord], iou_threshold: float) -> str:
    frames = [
        {
            "detections": [
                {"class": d.cls, "bbox": list(d.bbox), "confidence": d.confidence}
                for d in r.detections
            ],
            "ground_truth": [{"class": g.cls, "bbox": list(g.bbox)} for g in r.ground_truth],
        }
        for r in records
    ]
    return json.dumps({"iou_threshold": iou_threshold, "frames": frames})


def _parse_response(text: str) -> BatchMetrics:
    obj = json.loads(text)
    per_class = [
        ClassMetric(
            cls=c["class"],
            tp=c["tp"],
            fp=c["fp"],
            fn=c["fn"],
            precision=c["precision"],
            recall=c["recall"],
            f1=c["f1"],
            ap=c["ap"],
        )
        for c in obj["per_class"]
    ]
    return BatchMetrics(
        per_class=per_class,
        map=obj["map"],
        micro_precision=obj["micro_precision"],
        micro_recall=obj["micro_recall"],
        micro_f1=obj["micro_f1"],
    )


def compute_cpp(
    records: list[DetectionRecord],
    iou_threshold: float = 0.5,
    binary: str | None = None,
) -> BatchMetrics:
    """Compute batch metrics via the C++ aggregator subprocess."""
    path = binary or aggregator_path()
    if path is None:
        raise FileNotFoundError(
            "aggregator binary not found; set FLEETWATCH_AGGREGATOR or build it"
        )
    request = build_request(records, iou_threshold)
    proc = subprocess.run(
        [path],
        input=request,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"aggregator failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return _parse_response(proc.stdout)
