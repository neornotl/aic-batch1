"""Create a wide, reproducible BTC-keyframe audit plan for SOTUYEN1 Q11.

The public candidate was rejected by human review, so this deliberately does
not start from the submitted rows.  It combines several focused FTS searches
and keeps temporally distinct frames across many videos.  The resulting JSON
can be rendered by ``render_evidence_review.py``; every image it uses remains
an official BTC keyframe.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


QUESTION = (
    "Cảnh quay chậm tại vị trí vạch đích của cuộc đua xe đạp. Góc máy sát mặt đường "
    "bắt trọn khoảnh khắc về đích theo thứ tự nhất, nhì, ba lần lượt là 1 tay đua áo "
    "vàng quần đen, 1 tay đua áo xanh dương quần đen và 1 tay đua áo xanh dương quần đỏ"
)

# The captions are English and incomplete, so use several small searches.
# Each query puts race/cycling evidence first and only then adds a colour or
# finish-line clue.  This is intentionally different from the original broad
# retrieval query whose first candidate was rejected by manual review.
SEARCHES = (
    '"bicycle" AND "race"',
    '"cycling" AND "race"',
    '"cyclist" AND "race"',
    '"bicycle" AND "finish"',
    '"cyclist" AND "finish"',
    '"race" AND "yellow"',
    '"race" AND "blue"',
)


def rows_for_search(connection: sqlite3.Connection, expression: str, limit: int) -> list[dict[str, object]]:
    cursor = connection.execute(
        """SELECT k.keyframe_id, k.video_id, k.frame_id, k.keyframe_number, k.path,
                  k.caption, k.ocr, k.objects, bm25(keyframes_fts) AS score
           FROM keyframes_fts f JOIN keyframes k ON k.rowid=f.rowid
           WHERE keyframes_fts MATCH ? ORDER BY score LIMIT ?""",
        (expression, limit),
    )
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-search", type=int, default=800)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--per-video", type=int, default=3)
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    try:
        fused: dict[str, float] = defaultdict(float)
        selected_rows: dict[str, dict[str, object]] = {}
        provenance: dict[str, list[str]] = defaultdict(list)
        for expression in SEARCHES:
            for rank, row in enumerate(rows_for_search(connection, expression, args.per_search), 1):
                key = str(row["keyframe_id"])
                fused[key] += 1.0 / (60 + rank)
                selected_rows[key] = row
                provenance[key].append(expression)
    finally:
        connection.close()

    candidates: list[dict[str, object]] = []
    last_by_video: dict[str, int] = {}
    count_by_video: dict[str, int] = defaultdict(int)
    for key, score in sorted(fused.items(), key=lambda item: item[1], reverse=True):
        row = selected_rows[key]
        video_id = str(row["video_id"])
        number = int(row["keyframe_number"])
        if count_by_video[video_id] >= args.per_video:
            continue
        if video_id in last_by_video and abs(number - last_by_video[video_id]) < 8:
            continue
        count_by_video[video_id] += 1
        last_by_video[video_id] = number
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "video_id": video_id,
                "frames": [
                    {
                        "frame_id": int(row["frame_id"]),
                        "member": str(row["path"]).replace("\\", "/"),
                    }
                ],
                "score": round(score, 8),
                "searches": provenance[key],
                "caption": str(row.get("caption") or ""),
                "objects": str(row.get("objects") or ""),
            }
        )
        if len(candidates) >= args.candidates:
            break

    plan = {
        "version": 1,
        "audit": "Q11 independent FTS expansion after manual rejection of prior evidence",
        "queries": [
            {
                "id": "query-p1-11-kis",
                "kind": "kis",
                "text": QUESTION,
                "candidates": candidates,
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(candidates), "videos": len(count_by_video), "output": str(args.output)}))


if __name__ == "__main__":
    main()
