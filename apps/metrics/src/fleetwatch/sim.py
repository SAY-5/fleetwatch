"""Deterministic synthetic fleet detection generator.

Given a seed, the generator produces a reproducible stream of detection records
for N units across a sweep of operating conditions. Ground truth is laid down
first; detections are derived from it by a per-unit, per-condition quality model
so the resulting precision and recall are predictable and checkable.

The quality model is intentionally simple: every ground-truth box is detected
with probability ``recall_rate`` (a hit reproduces the GT box, so IoU is 1.0),
and a unit emits a number of spurious boxes governed by ``fp_rate``. A unit can
be made to degrade by lowering its recall over time, optionally only under a
chosen condition (used by the drift and attribution stages).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from .schema import (
    Condition,
    Detection,
    DetectionRecord,
    DistanceBand,
    GroundTruth,
    Lighting,
    Weather,
)

CLASSES = ["car", "pedestrian", "cyclist", "sign"]

_CONDITIONS = [
    Condition(lighting=lt, weather=w, distance_band=d)
    for lt in Lighting
    for w in Weather
    for d in DistanceBand
]


@dataclass
class DegradeSpec:
    """A scheduled degradation for one unit.

    ``per_window_drop`` is subtracted from the unit's recall once per window
    index (a window is ``frames_per_window`` consecutive frames). When
    ``only_condition`` is set, the drop applies solely to frames whose condition
    matches it, leaving other slices at the baseline recall. ``cliff_window``,
    when set, applies the full ``per_window_drop * 1`` step abruptly at that one
    window instead of gradually.
    """

    unit_id: str
    per_window_drop: float = 0.0
    only_condition: Condition | None = None
    cliff_window: int | None = None
    cliff_drop: float = 0.0


@dataclass
class FleetConfig:
    units: int = 4
    frames: int = 50
    seed: int = 7
    base_recall: float = 0.9
    base_fp_rate: float = 0.1
    frames_per_window: int = 10
    degrade: list[DegradeSpec] = field(default_factory=list)


def _unit_id(i: int) -> str:
    return f"unit-{i}"


def _effective_recall(cfg: FleetConfig, unit_id: str, window: int, cond: Condition) -> float:
    recall = cfg.base_recall
    for spec in cfg.degrade:
        if spec.unit_id != unit_id:
            continue
        if spec.only_condition is not None and spec.only_condition != cond:
            continue
        if spec.cliff_window is not None:
            if window >= spec.cliff_window:
                recall -= spec.cliff_drop
        else:
            recall -= spec.per_window_drop * window
    return max(0.0, min(1.0, recall))


def _gt_for_frame(rng: random.Random, n_objects: int) -> list[GroundTruth]:
    gts: list[GroundTruth] = []
    for _ in range(n_objects):
        cls = rng.choice(CLASSES)
        x1 = rng.uniform(0, 600)
        y1 = rng.uniform(0, 400)
        w = rng.uniform(20, 80)
        h = rng.uniform(20, 80)
        gts.append(GroundTruth.model_validate({"class": cls, "bbox": (x1, y1, x1 + w, y1 + h)}))
    return gts


def generate(cfg: FleetConfig) -> Iterator[DetectionRecord]:
    """Yield detection records for the whole fleet, deterministic for the seed."""
    rng = random.Random(cfg.seed)
    for u in range(cfg.units):
        unit_id = _unit_id(u)
        for frame in range(cfg.frames):
            cond = _CONDITIONS[(u + frame) % len(_CONDITIONS)]
            window = frame // cfg.frames_per_window
            recall = _effective_recall(cfg, unit_id, window, cond)

            n_objects = rng.randint(1, 5)
            gts = _gt_for_frame(rng, n_objects)

            detections: list[Detection] = []
            for gt in gts:
                if rng.random() < recall:
                    detections.append(
                        Detection.model_validate(
                            {
                                "class": gt.cls,
                                "bbox": gt.bbox,
                                "confidence": round(rng.uniform(0.5, 0.99), 4),
                            }
                        )
                    )
            # spurious detections
            n_fp = 1 if rng.random() < cfg.base_fp_rate else 0
            for _ in range(n_fp):
                cls = rng.choice(CLASSES)
                x1 = rng.uniform(0, 600)
                y1 = rng.uniform(0, 400)
                detections.append(
                    Detection.model_validate(
                        {
                            "class": cls,
                            "bbox": (x1, y1, x1 + 30, y1 + 30),
                            "confidence": round(rng.uniform(0.5, 0.99), 4),
                        }
                    )
                )

            yield DetectionRecord(
                unit_id=unit_id,
                frame_id=frame,
                timestamp=float(frame),
                condition=cond,
                detections=detections,
                ground_truth=gts,
            )
