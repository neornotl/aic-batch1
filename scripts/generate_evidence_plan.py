"""Build a small, portable keyframe-review plan from a candidate submission.

The plan contains the exact archive member for every selected prediction.  The
review workflow can therefore fetch just those JPEGs through HTTP byte ranges;
it never needs to download a Keyframes ZIP or a source video in full.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


def query_type(filename: str) -> str:
    value = Path(filename).stem.rsplit("-", 1)[-1].lower()
    if value not in {"kis", "qa", "trake"}:
        raise ValueError(f"Cannot determine query type: {filename}")
    return value


def selected_members(connection: sqlite3.Connection, wanted: set[tuple[str, int]]) -> dict[tuple[str, int], str]:
    """Read the table once instead of scanning it once per candidate frame."""
    result: dict[tuple[str, int], str] = {}
    for video_id, frame_id, path in connection.execute("SELECT video_id, frame_id, path FROM keyframes"):
        key = (str(video_id), int(frame_id))
        if key in wanted and key not in result:
            result[key] = str(path).replace("\\", "/")
    return result


def make_plan(
    submission_dir: Path,
    query_dir: Path,
    database: Path,
    per_query: int,
) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    try:
        raw_queries: list[tuple[Path, str, str, list[list[str]]]] = []
        wanted: set[tuple[str, int]] = set()
        for csv_path in sorted(submission_dir.glob("*.csv")):
            kind = query_type(csv_path.name)
            text_path = query_dir / f"{csv_path.stem}.txt"
            if not text_path.exists():
                raise FileNotFoundError(f"Missing query text: {text_path}")
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            chosen = rows[:per_query]
            for row in chosen:
                if len(row) < 2:
                    continue
                frame_columns = row[1:-1] if kind == "qa" else row[1:]
                for frame_id in frame_columns:
                    try:
                        wanted.add((row[0], int(frame_id)))
                    except ValueError:
                        continue
            raw_queries.append((csv_path, kind, text_path.read_text(encoding="utf-8-sig").strip(), chosen))

        members = selected_members(connection, wanted)
        queries: list[dict[str, Any]] = []
        for csv_path, kind, query_text, rows in raw_queries:
            candidates: list[dict[str, Any]] = []
            for rank, row in enumerate(rows, 1):
                if len(row) < 2:
                    continue
                video_id, frame_ids = row[0], row[1:]
                answer = frame_ids.pop() if kind == "qa" and frame_ids else None
                frames = []
                for frame_id in frame_ids:
                    member = members.get((video_id, int(frame_id)))
                    if member:
                        frames.append({"frame_id": int(frame_id), "member": member})
                if frames:
                    item: dict[str, Any] = {"rank": rank, "video_id": video_id, "frames": frames}
                    if answer is not None:
                        item["answer"] = answer
                    candidates.append(item)
            queries.append(
                {
                    "id": csv_path.stem,
                    "kind": kind,
                    "text": query_text,
                    "candidates": candidates,
                }
            )
    finally:
        connection.close()
    return {"version": 1, "queries": queries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=16)
    args = parser.parse_args()
    plan = make_plan(args.submission_dir, args.query_dir, args.database, args.per_query)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(candidate["frames"]) for query in plan["queries"] for candidate in query["candidates"])
    print(json.dumps({"queries": len(plan["queries"]), "frames": total, "output": str(args.output)}))


if __name__ == "__main__":
    main()
