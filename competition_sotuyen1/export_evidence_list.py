"""Export the top official frame candidates in a team-friendly Markdown list."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    markdown = ["# SOTUYEN1 — official evidence-frame shortlist", "",
                "Each frame_id is read from BTC metadata. Check the official Keyframes archive, not decoded video frames.", ""]
    csv_rows = [["query_id", "kind", "rank", "video_id", "frame_id", "query"]]
    for query in plan["queries"]:
        markdown.extend([f"## {query['id']}", "", query["text"], "", "| Rank | Video | Official frame_id |", "|---:|---|---:|"])
        for candidate in query["candidates"]:
            for frame in candidate["frames"]:
                markdown.append(f"| {candidate['rank']} | `{candidate['video_id']}` | `{frame['frame_id']}` |")
                csv_rows.append([query["id"], query["kind"], candidate["rank"], candidate["video_id"], frame["frame_id"], query["text"]])
        markdown.append("")
    args.markdown.write_text("\n".join(markdown), encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(csv_rows)
    print(f"wrote {args.markdown} and {args.csv}")


if __name__ == "__main__":
    main()
