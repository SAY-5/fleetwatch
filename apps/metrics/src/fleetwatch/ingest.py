"""Ingest of detection-record JSONL into in-memory records."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import DetectionRecord


def parse_line(line: str) -> DetectionRecord:
    """Parse one JSONL line into a DetectionRecord (raises on malformed input)."""
    return DetectionRecord.model_validate(json.loads(line))


def read_jsonl(path: Path) -> Iterator[DetectionRecord]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield parse_line(stripped)


def load(records: Iterable[str]) -> list[DetectionRecord]:
    return [parse_line(line) for line in records if line.strip()]


def cli() -> None:
    parser = argparse.ArgumentParser(description="Validate a detection-record JSONL file")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    count = sum(1 for _ in read_jsonl(args.path))
    print(f"validated {count} records in {args.path}")
