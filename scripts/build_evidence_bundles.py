"""Build compact, per-query review bundles from the official BTC archives.

Each bundle contains the exact official keyframes listed in
``competition_sotuyen1/EVIDENCE_FRAMES.csv`` and the source video of the
rank-1 candidate.  The full video is included for manual review, but frames
are never decoded from it: every JPEG is fetched by its published BTC member
path and renamed to the flattened evidence name (for example
``L22_V029_054.jpg``).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from evidence_review.range_zip import RangeZip


ARCHIVE_ROOT = "https://aic-data.ledo.io.vn/{}"
VIDEO_ARCHIVES = {
    f"L{number}": f"Videos_L{number}_a.zip" for number in range(21, 31)
}
QUERY_LABELS = {
    "query-p1-3-qa": "q3",
    "query-p1-11-kis": "q11",
    "query-p1-12-kis": "q12",
    "query-p1-13-kis": "q13",
    "query-p1-14-kis": "q14",
    "query-p1-15-qa": "q15",
}


def read_evidence(path: Path, requested: set[str]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = row["query_id"]
            if query_id in requested:
                grouped[query_id].append(row)
    missing = requested - set(grouped)
    if missing:
        raise ValueError(f"No evidence found for: {', '.join(sorted(missing))}")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return dict(grouped)


def write_readme(query_id: str, rows: list[dict[str, str]], output: Path) -> None:
    top = rows[0]
    lines = [
        f"# {QUERY_LABELS[query_id].upper()} review bundle",
        "",
        f"- Query ID: `{query_id}`",
        f"- Source video included: `videos/{top['video_id']}.mp4` (rank-1 evidence)",
        "- Frames were downloaded from the official BTC keyframe archive; no frames were extracted from video.",
        "",
        "## Exact evidence frames",
        "",
        "| Rank | File | frame_id | Official ZIP |",
        "|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | `frames/{row['keyframe_file']}.jpg` | "
            f"`{row['frame_id']}` | `{row['source_zip']}` |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_frames(rows: list[dict[str, str]], output: Path) -> None:
    archives: dict[str, RangeZip] = {}
    seen: set[str] = set()
    for row in rows:
        filename = f"{row['keyframe_file']}.jpg"
        if filename in seen:
            continue
        seen.add(filename)
        source_zip = row["source_zip"]
        archive = archives.setdefault(source_zip, RangeZip(ARCHIVE_ROOT.format(source_zip)))
        archive.download(row["member_path"], output / filename)


def download_rank_one_video(row: dict[str, str], output: Path) -> None:
    video_id = row["video_id"]
    prefix = video_id.split("_", 1)[0]
    try:
        archive_name = VIDEO_ARCHIVES[prefix]
    except KeyError as error:
        raise ValueError(f"No video archive mapping for {video_id}") from error
    archive = RangeZip(ARCHIVE_ROOT.format(archive_name), block_size=4 * 1024 * 1024)
    archive.download(video_id, output / f"{video_id}.mp4")


def build_query(query_id: str, rows: list[dict[str, str]], output: Path) -> dict[str, object]:
    label = QUERY_LABELS[query_id]
    bundle = output / label
    if bundle.exists():
        shutil.rmtree(bundle)
    frames = bundle / "frames"
    videos = bundle / "videos"
    frames.mkdir(parents=True)
    videos.mkdir()
    download_frames(rows, frames)
    download_rank_one_video(rows[0], videos)
    write_readme(query_id, rows, bundle / "README.md")
    return {
        "label": label,
        "query_id": query_id,
        "video_id": rows[0]["video_id"],
        "frames": len({row['keyframe_file'] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--queries",
        default=",".join(QUERY_LABELS),
        help="Comma-separated query IDs. Only Q3/Q11/Q12/Q13/Q14/Q15 are supported.",
    )
    args = parser.parse_args()
    requested = {item.strip() for item in args.queries.split(",") if item.strip()}
    unknown = requested - set(QUERY_LABELS)
    if unknown:
        raise ValueError(f"Unsupported query IDs: {', '.join(sorted(unknown))}")
    evidence = read_evidence(args.evidence, requested)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = [build_query(query_id, evidence[query_id], args.output) for query_id in sorted(requested)]
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
