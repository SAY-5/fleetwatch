# Benchmark

`make bench` runs the fleet-scale benchmark; `make bench-regress` runs it with a
30% regression gate against `bench/baseline.json`.

## What it measures

- ingest throughput (records parsed per second)
- Python reference `compute` time
- C++ aggregator end-to-end time (subprocess: encode, pipe, parse, compute)
- the resulting C++ speedup, plus the JSON vs binary request sizes

## Scales

| scale | units x frames | detections (approx) |
|-------|----------------|---------------------|
| smoke | 5 x 10         | ~140 (CI)           |
| small | 20 x 100       | ~5k                 |
| full  | 1000 x 1000    | ~2.75M              |

## Findings

The protocol dominates at scale. The original JSON transport made the C++
aggregator slower end-to-end than pure Python, because shuttling and text-parsing
a multi-hundred-MB request cost far more than the metric compute. Two changes fix
this:

1. a raw block read of stdin on the C++ side (a per-character stream iterator is
   O(n) with heavy overhead), and
2. a compact little-endian binary request protocol (FWB1) that is parse-free and
   roughly a third the size of the JSON.

At the full scale (~2.75M detections) the C++ aggregator with the binary protocol
runs about **1.57x** faster than the Python reference (about 537k vs 342k
detections per second), and the request shrinks from ~721 MB of JSON to ~226 MB
of binary. At small scales the subprocess spawn overhead dominates and the
speedup is below 1x, which is expected: the C++ path only pays off once the
batch is large.

## Regression gate

`make bench-regress` computes the current speedup and compares it to the stored
baseline. It fails when the speedup falls below `baseline * (1 - 0.30)`. The
first run writes the baseline. CI runs only the smoke-scale bench tests to keep
the pipeline fast; the full bench is a local tool.
