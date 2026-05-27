"""Tests for the deterministic synthetic fleet generator."""

from __future__ import annotations

from fleetwatch.schema import Condition, DistanceBand, Lighting, Weather
from fleetwatch.sim import DegradeSpec, FleetConfig, generate


def test_generator_is_deterministic_for_a_seed() -> None:
    cfg = FleetConfig(units=3, frames=20, seed=11)
    a = [r.model_dump_json(by_alias=True) for r in generate(cfg)]
    b = [r.model_dump_json(by_alias=True) for r in generate(cfg)]
    assert a == b


def test_different_seeds_diverge() -> None:
    a = [r.model_dump_json() for r in generate(FleetConfig(units=2, frames=10, seed=1))]
    b = [r.model_dump_json() for r in generate(FleetConfig(units=2, frames=10, seed=2))]
    assert a != b


def test_record_counts_match_units_times_frames() -> None:
    cfg = FleetConfig(units=4, frames=25, seed=3)
    recs = list(generate(cfg))
    assert len(recs) == 4 * 25


def test_hits_reproduce_ground_truth_boxes_exactly() -> None:
    # With full recall and no false positives every detection should coincide
    # with a ground-truth box, which makes downstream IoU exactly 1.0.
    cfg = FleetConfig(units=1, frames=30, seed=5, base_recall=1.0, base_fp_rate=0.0)
    for rec in generate(cfg):
        gt_boxes = {(g.cls, g.bbox) for g in rec.ground_truth}
        for det in rec.detections:
            assert (det.cls, det.bbox) in gt_boxes


def test_perfect_unit_detects_every_ground_truth() -> None:
    cfg = FleetConfig(units=1, frames=40, seed=9, base_recall=1.0, base_fp_rate=0.0)
    for rec in generate(cfg):
        assert len(rec.detections) == len(rec.ground_truth)


def test_night_only_degradation_leaves_day_frames_untouched() -> None:
    night = Condition(
        lighting=Lighting.NIGHT, weather=Weather.CLEAR, distance_band=DistanceBand.NEAR
    )
    cfg = FleetConfig(
        units=1,
        frames=200,
        seed=4,
        base_recall=1.0,
        base_fp_rate=0.0,
        degrade=[DegradeSpec(unit_id="unit-0", per_window_drop=0.1, only_condition=night)],
    )
    for rec in generate(cfg):
        if rec.condition != night:
            # untouched conditions keep full recall: detections == ground truth
            assert len(rec.detections) == len(rec.ground_truth)
