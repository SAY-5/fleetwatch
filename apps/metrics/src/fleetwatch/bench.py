"""Fleet-scale benchmark: C++ aggregator throughput vs the Python reference.

Generates a synthetic fleet, measures detections-per-second and mAP-compute time
for both the C++ aggregator and the pure-Python reference, reports the speedup,
and measures ingest (JSON parse) throughput. A regression gate compares the
measured C++ speedup against a stored baseline and fails when it drops by more
than a tolerance (default 30%).

Scales:
  smoke  ~5k detections   (CI)
  small  ~100k detections
  full   ~10M detections   (1000 units x 1000 frames x ~10 detections)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import metrics_ref
from .aggregator import (
    aggregator_path,
    build_binary_request,
    build_request,
    compute_cpp,
)
from .schema import DetectionRecord
from .sim import FleetConfig, generate

SCALES: dict[str, tuple[int, int]] = {
    "smoke": (5, 10),
    "small": (20, 100),
    "full": (1000, 1000),
}

_BASELINE = Path(__file__).resolve().parents[4] / "bench" / "baseline.json"


@dataclass
class BenchResult:
    scale: str
    units: int
    frames: int
    n_records: int
    n_detections: int
    json_bytes: int
    binary_bytes: int
    ingest_sec: float
    ingest_per_sec: float
    python_sec: float
    cpp_sec: float
    python_det_per_sec: float
    cpp_det_per_sec: float
    speedup: float


def _make_records(units: int, frames: int, seed: int = 7) -> list[DetectionRecord]:
    cfg = FleetConfig(units=units, frames=frames, seed=seed, base_recall=0.85, base_fp_rate=0.2)
    return list(generate(cfg))


def _time_ingest(records: list[DetectionRecord]) -> tuple[float, int]:
    payloads = [r.model_dump_json(by_alias=True) for r in records]
    start = time.perf_counter()
    parsed = [DetectionRecord.model_validate_json(p) for p in payloads]
    elapsed = time.perf_counter() - start
    return elapsed, len(parsed)


def run(scale: str, seed: int = 7) -> BenchResult:
    units, frames = SCALES[scale]
    records = _make_records(units, frames, seed)
    n_det = sum(len(r.detections) for r in records)

    ingest_sec, _ = _time_ingest(records)

    json_bytes = len(build_request(records, 0.5))
    binary_bytes = len(build_binary_request(records, 0.5))

    t0 = time.perf_counter()
    metrics_ref.compute(records, 0.5)
    python_sec = time.perf_counter() - t0

    if aggregator_path() is None:
        cpp_sec = float("nan")
        speedup = float("nan")
        cpp_det_per_sec = float("nan")
    else:
        t1 = time.perf_counter()
        compute_cpp(records, 0.5, wire="binary")
        cpp_sec = time.perf_counter() - t1
        speedup = python_sec / cpp_sec if cpp_sec > 0 else float("inf")
        cpp_det_per_sec = n_det / cpp_sec if cpp_sec > 0 else float("inf")

    return BenchResult(
        scale=scale,
        units=units,
        frames=frames,
        n_records=len(records),
        n_detections=n_det,
        json_bytes=json_bytes,
        binary_bytes=binary_bytes,
        ingest_sec=ingest_sec,
        ingest_per_sec=len(records) / ingest_sec if ingest_sec > 0 else float("inf"),
        python_sec=python_sec,
        cpp_sec=cpp_sec,
        python_det_per_sec=n_det / python_sec if python_sec > 0 else float("inf"),
        cpp_det_per_sec=cpp_det_per_sec,
        speedup=speedup,
    )


def _print(result: BenchResult) -> None:
    print(json.dumps(asdict(result), indent=2))
    print(
        f"\n[{result.scale}] {result.n_detections} detections | "
        f"python {result.python_sec:.3f}s, cpp {result.cpp_sec:.3f}s | "
        f"speedup {result.speedup:.2f}x | "
        f"ingest {result.ingest_per_sec:,.0f} rec/s"
    )


def regress(tolerance: float, scale: str = "small") -> int:
    """Compare the current speedup against the stored baseline; return exit code."""
    result = run(scale)
    _print(result)
    if not _BASELINE.exists():
        _BASELINE.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE.write_text(json.dumps({"scale": scale, "speedup": result.speedup}, indent=2))
        print(f"\nno baseline; wrote {_BASELINE}")
        return 0
    baseline = json.loads(_BASELINE.read_text())
    base_speedup = float(baseline["speedup"])
    floor = base_speedup * (1.0 - tolerance)
    print(f"\nbaseline speedup {base_speedup:.2f}x, floor {floor:.2f}x, now {result.speedup:.2f}x")
    if result.speedup < floor:
        print("REGRESSION: speedup dropped beyond tolerance")
        return 1
    print("OK: within tolerance")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FleetWatch fleet-scale benchmark")
    parser.add_argument("--scale", choices=sorted(SCALES), default="small")
    parser.add_argument(
        "--regress", type=float, default=None, help="regression tolerance, e.g. 0.30"
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.regress is not None:
        return regress(args.regress, args.scale)
    _print(run(args.scale, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
