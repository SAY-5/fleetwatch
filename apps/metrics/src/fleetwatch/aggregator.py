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
import struct
import subprocess
from pathlib import Path

from .metrics_ref import BatchMetrics, ClassMetric
from .schema import DetectionRecord

_ENV_VAR = "FLEETWATCH_AGGREGATOR"
_MAGIC = b"FWB1"


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


def build_binary_request(records: list[DetectionRecord], iou_threshold: float) -> bytes:
    """Encode the batch in the compact little-endian binary protocol (FWB1).

    Roughly a fifth the size of the JSON request and parse-free on the C++ side,
    so it removes the transport cost that dominates JSON at fleet scale.
    """
    # Build a class-name table; detections and ground truth reference it by id.
    classes: dict[str, int] = {}
    for r in records:
        for d in r.detections:
            if d.cls not in classes:
                classes[d.cls] = len(classes)
        for g in r.ground_truth:
            if g.cls not in classes:
                classes[g.cls] = len(classes)

    parts: list[bytes] = [_MAGIC, struct.pack("<d", iou_threshold)]
    parts.append(struct.pack("<I", len(classes)))
    for name in classes:
        encoded = name.encode("utf-8")
        parts.append(struct.pack("<H", len(encoded)))
        parts.append(encoded)

    parts.append(struct.pack("<I", len(records)))
    for r in records:
        parts.append(struct.pack("<II", len(r.detections), len(r.ground_truth)))
        for d in r.detections:
            parts.append(struct.pack("<H", classes[d.cls]))
            parts.append(struct.pack("<5d", *d.bbox, d.confidence))
        for g in r.ground_truth:
            parts.append(struct.pack("<H", classes[g.cls]))
            parts.append(struct.pack("<4d", *g.bbox))
    return b"".join(parts)


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
    binary_path: str | None = None,
    wire: str = "binary",
    env: dict[str, str] | None = None,
) -> BatchMetrics:
    """Compute batch metrics via the C++ aggregator subprocess.

    ``wire`` selects the request encoding: ``"binary"`` (the compact FWB1 format,
    default) or ``"json"``. The binary format removes the transport cost that
    dominates JSON at fleet scale; both produce identical metrics.
    """
    path = binary_path or aggregator_path()
    if path is None:
        raise FileNotFoundError(
            "aggregator binary not found; set FLEETWATCH_AGGREGATOR or build it"
        )
    if wire == "binary":
        request = build_binary_request(records, iou_threshold)
    elif wire == "json":
        request = build_request(records, iou_threshold).encode("utf-8")
    else:
        raise ValueError(f"unknown wire format {wire}")
    proc = subprocess.run(
        [path],
        input=request,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"aggregator failed (rc={proc.returncode}): {proc.stderr.decode('utf-8').strip()}"
        )
    return _parse_response(proc.stdout.decode("utf-8"))
