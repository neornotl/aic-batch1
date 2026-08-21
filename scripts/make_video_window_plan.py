"""Create an official-keyframe review plan for one contiguous video window."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start > args.end:
        raise ValueError("start must not be after end")
    connection = sqlite3.connect(args.database)
    try:
        rows = connection.execute(
            """SELECT keyframe_number, frame_id, path FROM keyframes
               WHERE video_id=? AND keyframe_number BETWEEN ? AND ?
               ORDER BY keyframe_number""",
            (args.video, args.start, args.end),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError("no matching official keyframes")
    candidates = [
        {
            "rank": rank,
            "video_id": args.video,
            "frames": [{"frame_id": int(frame_id), "member": str(path).replace("\\", "/")}],
        }
        for rank, (_, frame_id, path) in enumerate(rows, 1)
    ]
    plan = {"version": 1, "queries": [{"id": args.query_id, "kind": "qa", "text": args.text, "candidates": candidates}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(candidates), "output": str(args.output)}))


if __name__ == "__main__":
    main()
