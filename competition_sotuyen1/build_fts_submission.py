"""Build a review-only CSV submission directory from official retrieval rows."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists() or args.query_output.exists():
        raise SystemExit("refusing to overwrite an existing competition directory")
    data = json.loads(args.retrieval.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    args.query_output.mkdir(parents=True)
    with zipfile.ZipFile(args.bundle) as bundle:
        for query_name, item in sorted(data.items()):
            query_stem = Path(query_name).stem
            kind = query_stem.rsplit("-", 1)[-1]
            (args.query_output / query_name).write_bytes(bundle.read(query_name))
            unique: set[tuple[str, int]] = set()
            selected: list[tuple[str, int]] = []
            for row in item["ranked_frames"]:
                candidate = (str(row["video_id"]), int(row["frame_id"]))
                if candidate not in unique:
                    unique.add(candidate)
                    selected.append(candidate)
                if len(selected) == args.rows:
                    break
            if not selected:
                raise SystemExit(f"no rows for {query_name}")
            with (args.output / f"{query_stem}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for video_id, frame_id in selected:
                    # QA answers are intentionally a visible placeholder until
                    # official keyframe review resolves the value.
                    writer.writerow([video_id, frame_id] + (["PENDING"] if kind == "qa" else []))
    print(f"created {args.output} and {args.query_output}")


if __name__ == "__main__":
    main()
