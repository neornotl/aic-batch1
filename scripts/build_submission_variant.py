#!/usr/bin/env python3
"""Create a non-destructive submission variant from verified overrides.

The source directory is never modified.  Each overridden query receives the
verified rows first, followed by its previous candidates without duplicates.
The original candidate count is preserved so the generated directory has the
same shape as its source and can be inspected or zipped later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle) if row]


def write_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def merge_rows(original: list[list[str]], overrides: list[list[str]]) -> list[list[str]]:
    if len(overrides) > len(original):
        raise ValueError("override has more rows than the source query")
    merged: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in overrides + original:
        key = tuple(row)
        if key not in seen:
            merged.append(row)
            seen.add(key)
        if len(merged) == len(original):
            return merged
    raise ValueError("not enough unique rows to preserve the source query size")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if not args.source.is_dir():
        raise SystemExit(f"source is not a directory: {args.source}")

    overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    source_files = sorted(args.source.glob("*.csv"))
    source_names = {path.name for path in source_files}
    unknown = set(overrides) - source_names
    if unknown:
        raise SystemExit(f"overrides refer to missing source files: {sorted(unknown)}")

    args.output.mkdir(parents=True)
    changed = 0
    for source_path in source_files:
        original = read_rows(source_path)
        selected = merge_rows(original, overrides[source_path.name]) if source_path.name in overrides else original
        write_rows(args.output / source_path.name, selected)
        changed += source_path.name in overrides

    print(f"created {args.output} with {len(source_files)} CSVs; {changed} queries overridden")


if __name__ == "__main__":
    main()
