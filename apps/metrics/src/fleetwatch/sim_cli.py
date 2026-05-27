"""CLI for the synthetic fleet generator: writes detection records as JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from .sim import FleetConfig, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic fleet detection run")
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = FleetConfig(units=args.units, frames=args.frames, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in generate(cfg):
            fh.write(rec.model_dump_json(by_alias=True))
            fh.write("\n")
            count += 1
    print(f"wrote {count} records to {args.out}")


if __name__ == "__main__":
    main()
