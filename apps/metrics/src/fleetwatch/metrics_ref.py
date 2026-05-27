"""Pure-Python reference implementation of the detection metrics.

This module is the oracle: the C++ aggregator must agree with it to within a
documented tolerance. It therefore mirrors the C++ algorithms exactly, including
sort order and accumulation order. See ``docs/metrics.md`` for the definitions
and ``docs/aggregator-protocol.md`` for the wire format.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import BBox, DetectionRecord


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes; symmetric, 0.0 when disjoint."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    uni = area_a + area_b - inter
    if uni <= 0.0:
        return 0.0
    return inter / uni


@dataclass
class ScoredDet:
    confidence: float
    is_tp: bool
    cls: str


@dataclass
class ClassMetric:
    cls: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap: float


@dataclass
class BatchMetrics:
    per_class: list[ClassMetric]
    map: float
    micro_precision: float
    micro_recall: float
    micro_f1: float


def _match_frame(
    record: DetectionRecord, iou_threshold: float
) -> tuple[list[ScoredDet], dict[str, int]]:
    """Greedy per-class matching for one frame, mirroring the C++ ``match``."""
    dets = list(record.detections)
    # confidence descending, stable on original index
    order = sorted(range(len(dets)), key=lambda i: -dets[i].confidence)

    gts = list(record.ground_truth)
    claimed = [False] * len(gts)
    scored: list[ScoredDet] = []

    for oi in order:
        d = dets[oi]
        best_iou = iou_threshold
        best_gt = -1
        for gi, gt in enumerate(gts):
            if claimed[gi] or gt.cls != d.cls:
                continue
            v = iou(d.bbox, gt.bbox)
            if v >= best_iou:
                best_iou = v
                best_gt = gi
        is_tp = best_gt >= 0
        if is_tp:
            claimed[best_gt] = True
        scored.append(ScoredDet(confidence=d.confidence, is_tp=is_tp, cls=d.cls))

    gt_per_class: dict[str, int] = {}
    for gt in gts:
        gt_per_class[gt.cls] = gt_per_class.get(gt.cls, 0) + 1
    return scored, gt_per_class


def _pr_points(scored: list[ScoredDet], n_gt: int) -> list[tuple[float, float]]:
    """Return (recall, precision) points in confidence-descending order."""
    ordered = sorted(scored, key=lambda s: (-s.confidence, not s.is_tp))
    points: list[tuple[float, float]] = []
    tp = 0
    fp = 0
    for s in ordered:
        if s.is_tp:
            tp += 1
        else:
            fp += 1
        denom = tp + fp
        precision = tp / denom if denom > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        points.append((recall, precision))
    return points


def average_precision(points: list[tuple[float, float]], n_gt: int) -> float:
    """All-points AP: area under the PR curve with a non-increasing envelope."""
    if n_gt <= 0 or not points:
        return 0.0
    rec = [0.0]
    prec = [1.0]
    for r, p in points:
        rec.append(r)
        prec.append(p)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for i in range(1, len(rec)):
        dr = rec[i] - rec[i - 1]
        if dr > 0.0:
            ap += dr * prec[i]
    return ap


def compute(records: list[DetectionRecord], iou_threshold: float = 0.5) -> BatchMetrics:
    """Aggregate a batch of records into per-class and overall metrics."""
    scored_by_class: dict[str, list[ScoredDet]] = {}
    gt_count: dict[str, int] = {}

    for rec in records:
        scored, gpc = _match_frame(rec, iou_threshold)
        for s in scored:
            scored_by_class.setdefault(s.cls, []).append(s)
        for cls, n in gpc.items():
            gt_count[cls] = gt_count.get(cls, 0) + n

    classes = sorted(set(scored_by_class) | set(gt_count))

    per_class: list[ClassMetric] = []
    ap_sum = 0.0
    total_tp = total_fp = total_fn = 0

    for cls in classes:
        dets = scored_by_class.get(cls, [])
        n_gt = gt_count.get(cls, 0)
        tp = sum(1 for s in dets if s.is_tp)
        fp = sum(1 for s in dets if not s.is_tp)
        fn = n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        ap = average_precision(_pr_points(dets, n_gt), n_gt)
        per_class.append(ClassMetric(cls, tp, fp, fn, precision, recall, f1, ap))
        ap_sum += ap
        total_tp += tp
        total_fp += fp
        total_fn += fn

    mapv = ap_sum / len(classes) if classes else 0.0
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return BatchMetrics(per_class, mapv, micro_p, micro_r, micro_f1)
