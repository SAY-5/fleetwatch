"""Detection-record schema shared across ingest, metrics, sim and the C++ protocol.

A detection record is one frame reported by one fleet unit. It carries the
operating condition, the model's detections (bbox + class + confidence) and the
ground-truth labels for that frame. Bounding boxes are ``[x1, y1, x2, y2]`` in
pixel coordinates with ``x2 > x1`` and ``y2 > y1``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Lighting(StrEnum):
    DAY = "day"
    NIGHT = "night"
    DUSK = "dusk"


class Weather(StrEnum):
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"


class DistanceBand(StrEnum):
    NEAR = "near"
    MID = "mid"
    FAR = "far"


class Condition(BaseModel):
    lighting: Lighting
    weather: Weather
    distance_band: DistanceBand

    model_config = {"frozen": True}

    def slice_key(self) -> str:
        """Stable string key used for condition slicing in the store."""
        return f"{self.lighting.value}|{self.weather.value}|{self.distance_band.value}"


BBox = tuple[float, float, float, float]


class Detection(BaseModel):
    """A predicted box with a class label and a confidence in [0, 1]."""

    cls: str = Field(alias="class")
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}

    @field_validator("bbox")
    @classmethod
    def _valid_box(cls, v: BBox) -> BBox:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
        return v


class GroundTruth(BaseModel):
    """A ground-truth box with a class label."""

    cls: str = Field(alias="class")
    bbox: BBox

    model_config = {"populate_by_name": True}

    @field_validator("bbox")
    @classmethod
    def _valid_box(cls, v: BBox) -> BBox:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
        return v


class DetectionRecord(BaseModel):
    """One frame from one unit: detections plus ground truth under a condition."""

    unit_id: str
    frame_id: int = Field(ge=0)
    timestamp: float
    condition: Condition
    detections: list[Detection] = Field(default_factory=list)
    ground_truth: list[GroundTruth] = Field(default_factory=list)

    model_config = {"populate_by_name": True}
