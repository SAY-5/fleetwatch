# FleetWatch

Detection performance metrics and dashboard system for a simulated robot fleet.

FleetWatch ingests detection and recognition results from a fleet of units. Each
unit reports per-frame detections (bounding box, class, confidence) along with
ground-truth labels. FleetWatch computes precision, recall and mAP, tracks drift
over time, and surfaces dashboards plus trend alerts that flag which units and
which operating conditions (lighting, weather, distance) degraded most.

## What this is and is not

FleetWatch measures **computer-vision detection quality** over a fleet. It is not
a system-health monitor and not an LLM-eval tool. The signal is precision / recall
/ mAP / drift on object detections.

## Stack

- Python 3.12 for ingest, metrics orchestration, the FastAPI dashboard and alerts.
- C++20 for a fast per-batch metric aggregator (IoU matching, PR curve, mAP),
  invoked from Python over a subprocess JSON protocol.
- PostgreSQL for the metric store (via testcontainers in CI).
- Pydantic, structlog, pytest, hypothesis, testcontainers.
- CMake + GoogleTest for the C++ side.

## Layout

```
apps/metrics/      Python: ingest, metrics_ref, drift, alerts, api, store
apps/aggregator/   C++: iou, match, prcurve, map, main + GoogleTest
sim/               synthetic fleet detection generator
examples/runs/     example detection runs (jsonl)
docs/              metrics, drift, conditions, aggregator protocol
```

## Quick start

```bash
make setup        # install python deps via poetry
make build-cpp    # configure + build the C++ aggregator
make test         # python + cpp unit tests
make sim          # generate an example fleet run
make api          # serve the dashboard at :8000
```

See `docs/` for the metric definitions, drift model, condition slicing and the
Python to C++ aggregator protocol.

## License

MIT. See [LICENSE](LICENSE).
