"""Download one official BTC video via ZIP byte ranges, including ZIP64 archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from evidence_review.range_zip import RangeZip


ARCHIVE_ROOT = "https://aic-data.ledo.io.vn/Videos_{}_a.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="e.g. L26_V041")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    video_id = args.video.strip()
    prefix = video_id.split("_", 1)[0]
    if prefix not in {f"L{number}" for number in range(21, 31)}:
        raise ValueError(f"unsupported BTC video id: {video_id}")
    archive = RangeZip(ARCHIVE_ROOT.format(prefix), block_size=4 * 1024 * 1024)
    target = args.output / f"{video_id}.mp4"
    archive.download(video_id, target)
    print(f"{video_id} {target.stat().st_size}")


if __name__ == "__main__":
    main()
