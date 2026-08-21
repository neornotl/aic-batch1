"""Export the top official frame candidates in a team-friendly Markdown list."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def source_zip(video_id: str) -> str:
    """Return the BTC Keyframes archive that contains ``video_id``."""
    batch, number_text = video_id.split("_V", 1)
    if batch != "L26":
        return f"Keyframes_{batch}.zip"
    number = int(number_text)
    if not 1 <= number <= 499:
        raise ValueError(f"unexpected L26 video number: {video_id}")
    suffix = chr(ord("a") + (number - 1) // 100)
    return f"Keyframes_L26_{suffix}.zip"


def keyframe_file_name(video_id: str, member: str) -> str:
    """Use the name teammates see after opening the BTC Keyframes folders."""
    return f"{video_id}_{Path(member).stem}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    markdown = ["# SOTUYEN1 — official evidence-frame shortlist", "",
                "Each frame_id is read from BTC metadata. Check the official Keyframes archive, not decoded video frames.", ""]
    markdown[1:1] = [
        "Use **Keyframe file** to locate the image in the BTC Keyframes folders. "
        "`Source ZIP` identifies the required archive; L26 is split across a-e.",
        "",
    ]
    csv_rows = [["query_id", "kind", "rank", "keyframe_file", "video_id", "frame_id", "source_zip", "member_path", "query"]]
    for query in plan["queries"]:
        query_text = query["text"].replace(" \n", "\n").rstrip()
        markdown.extend([f"## {query['id']}", "", query_text, "",
                         "| Rank | Keyframe file | Official frame_id | Source ZIP |",
                         "|---:|---|---:|---|"])
        for candidate in query["candidates"]:
            for frame in candidate["frames"]:
                video_id = candidate["video_id"]
                member = frame["member"]
                name = keyframe_file_name(video_id, member)
                archive = source_zip(video_id)
                markdown.append(f"| {candidate['rank']} | `{name}` | `{frame['frame_id']}` | `{archive}` |")
                csv_rows.append([query["id"], query["kind"], candidate["rank"], name, video_id,
                                 frame["frame_id"], archive, member, query_text])
        markdown.append("")
    args.markdown.write_text("\n".join(markdown), encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(csv_rows)
    print(f"wrote {args.markdown} and {args.csv}")


if __name__ == "__main__":
    main()
