# Aggregator protocol

The C++ aggregator (`fleetwatch-aggregator`) is a stateless filter. Python sends
one JSON request on stdin and reads one JSON response on stdout. There is no
streaming and no persistent state, which keeps it trivial to invoke from a
subprocess and trivial to test hermetically.

## Request

```json
{
  "iou_threshold": 0.5,
  "frames": [
    {
      "detections": [
        {"class": "car", "bbox": [0, 0, 10, 10], "confidence": 0.9}
      ],
      "ground_truth": [
        {"class": "car", "bbox": [0, 0, 10, 10]}
      ]
    }
  ]
}
```

- `iou_threshold` is optional and defaults to `0.5`. It must be in `[0, 1]`.
- `frames` is required (may be empty).
- `bbox` is `[x1, y1, x2, y2]` with `x2 > x1` and `y2 > y1`.
- `confidence` is optional per detection and defaults to `1.0`.

Malformed input (truncated JSON, bad box, unknown field, trailing garbage)
causes the process to exit with code `2` and a message on stderr. This is the
contract the JSON fuzz test exercises on both the Python and C++ sides.

## Response

```json
{
  "iou_threshold": 0.5,
  "map": 1.0,
  "micro_precision": 1.0,
  "micro_recall": 1.0,
  "micro_f1": 1.0,
  "per_class": [
    {"class": "car", "tp": 1, "fp": 0, "fn": 0,
     "precision": 1.0, "recall": 1.0, "f1": 1.0, "ap": 1.0}
  ]
}
```

Floats are written with 12 significant digits (`%.12g`). The Python reference
and the C++ aggregator agree to within `1e-9` on every field; see
[metrics.md](metrics.md) for why the tolerance is not exactly zero.

## Binary protocol (FWB1)

For fleet-scale batches the JSON request is replaced by a compact little-endian
binary protocol. The aggregator auto-detects it by the 4-byte magic `FWB1` at
the start of stdin and falls back to JSON otherwise, so both paths share one
binary. The binary layout is a header (`magic`, `f64 iou_threshold`, a class-name
table) followed by frames of fixed-width detection and ground-truth records; see
`apps/aggregator/src/binproto.h`. It carries the same information in roughly a
third of the bytes with no text parsing, which is what makes the C++ aggregator
faster than the Python reference end-to-end at scale (see
[bench.md](bench.md)). The Python client (`compute_cpp`) uses the binary wire by
default and can be forced back to JSON with `wire="json"`; both produce identical
metrics.

## Determinism

The aggregator is deterministic for a given request. Matching sorts detections
by confidence descending with stable tie-breaking on original index; the PR
curve sorts by confidence descending with true positives before false positives
on ties. Both sides use the same accumulation order so the metrics match.
