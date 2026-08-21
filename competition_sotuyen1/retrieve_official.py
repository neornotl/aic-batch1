"""Rank official frame records for the SOTUYEN1 query bundle.

This reads only the local caption/index database.  `frame_id` in every result
comes directly from the metadata-backed keyframes table; it is never inferred
from video decoding or a timestamp.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import zipfile
from collections import defaultdict
from pathlib import Path


TERMS = {
    "query-p1-1-kis.txt": "exercise group people hands toes glasses red hats",
    "query-p1-2-kis.txt": "map dam aerial rain",
    "query-p1-3-qa.txt": "fish weighing scale number",
    "query-p1-4-kis.txt": "lion London Zoo keeper green weighing",
    "query-p1-5-kis.txt": "peas squid wok onion red pepper flame",
    "query-p1-6-kis.txt": "man suit gemstone woman headscarf open pit mine",
    "query-p1-7-kis.txt": "star carrot boiling basket chopsticks okra broccoli pink sauce",
    "query-p1-8-kis.txt": "chef sticks flower slices steaming plate chopsticks spoon glass bowl",
    "query-p1-9-qa.txt": "cars flood yellow red black bridge sign",
    "query-p1-10-kis.txt": "grape black scissors blue string vine",
    "query-p1-11-kis.txt": "bicycle race finish yellow black blue red",
    "query-p1-12-kis.txt": "motorbike rider petrol station fuel oil price",
    "query-p1-13-kis.txt": "underwater flashlight fishing net dawn cameramen",
    "query-p1-14-kis.txt": "chef sticks flower slices steaming plate chopsticks spoon glass bowl",
    "query-p1-15-qa.txt": "earthquake map legend magnitude",
}


def fts_expression(terms: str) -> str:
    return " OR ".join(f'"{term}"' for term in terms.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=160)
    args = parser.parse_args()

    with zipfile.ZipFile(args.bundle) as bundle:
        query_names = sorted(name for name in bundle.namelist() if name.endswith(".txt"))
        missing = set(query_names) - set(TERMS)
        if missing:
            raise SystemExit(f"missing retrieval terms for: {sorted(missing)}")
        queries = {name: bundle.read(name).decode("utf-8").strip() for name in query_names}

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    results: dict[str, dict] = {}
    for name, question in queries.items():
        cursor = connection.execute(
            """SELECT k.video_id, k.frame_id, k.keyframe_number, k.caption, k.ocr, k.objects,
                      bm25(keyframes_fts) AS score
               FROM keyframes_fts f JOIN keyframes k ON k.rowid = f.rowid
               WHERE keyframes_fts MATCH ? ORDER BY score LIMIT ?""",
            (fts_expression(TERMS[name]), args.limit),
        )
        rows = [dict(row) for row in cursor]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["video_id"]].append(row)
        # Keep representative frames for the strongest distinct video ids.
        videos = [
            {"video_id": video_id, "hits": len(items), "frames": items[:8]}
            for video_id, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[1][0]["score"]))
        ]
        results[name] = {
            "question": question,
            "terms": TERMS[name],
            "ranked_frames": rows,
            "videos": videos,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output} for {len(results)} queries")


if __name__ == "__main__":
    main()
