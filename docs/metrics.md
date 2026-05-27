# Detection metrics

FleetWatch computes the standard object-detection metrics at a single IoU
threshold (default `0.5`). The Python reference in `fleetwatch.metrics_ref` is
the oracle; the C++ aggregator reproduces it.

## IoU

For boxes `a` and `b` given as `[x1, y1, x2, y2]`:

```
inter = max(0, min(ax2,bx2) - max(ax1,bx1)) * max(0, min(ay2,by2) - max(ay1,by1))
union = area(a) + area(b) - inter
IoU   = inter / union   (0 when union <= 0)
```

IoU is symmetric and lies in `[0, 1]`.

## Matching

Per frame, per class, greedy matching:

1. Sort detections by confidence descending; ties keep the original index order
   (a stable sort).
2. Each detection claims the highest-IoU unclaimed ground-truth box of the same
   class whose IoU is at least the threshold. A detection that claims a box is a
   true positive (TP); otherwise it is a false positive (FP).
3. Ground-truth boxes left unclaimed are false negatives (FN).

## Precision, recall, F1

Per class, accumulated over all frames in the batch:

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)          where TP + FN = number of GT boxes
F1        = 2 * P * R / (P + R)
```

Each is `0.0` when its denominator is `0`. Micro-averaged variants pool TP/FP/FN
across all classes.

## Average precision and mAP

AP is the area under the precision-recall curve using the all-points
(continuous) method: build the PR curve over detections sorted by confidence
descending (ties put TPs first), prepend the `(recall=0, precision=1)` sentinel,
make precision monotonically non-increasing from the right, and integrate
precision over the recall axis. mAP is the unweighted mean of per-class AP over
the union of classes that appear as a detection or as ground truth.

## Python / C++ parity tolerance

The two implementations use the same algorithm, the same sort order and the same
accumulation order, so they agree to within `1e-9` on every field. The tolerance
is not exactly zero because summation happens in IEEE-754 double precision and
the JSON transport serialises floats with 12 significant digits. The parity
property test in the test suite asserts the `1e-9` bound on random batches.
