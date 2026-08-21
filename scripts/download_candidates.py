from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from aic_pipeline.range_zip import RangeZip


ARCHIVES = {
    "L21": "https://aic-data.ledo.io.vn/Videos_L21_a.zip",
    "L22": "https://aic-data.ledo.io.vn/Videos_L22_a.zip",
    "L23": "https://aic-data.ledo.io.vn/Videos_L23_a.zip",
    "L24": "https://aic-data.ledo.io.vn/Videos_L24_a.zip",
    "L25": "https://aic-data.ledo.io.vn/Videos_L25_a.zip",
    "L26": "https://aic-data.ledo.io.vn/Videos_L26_a.zip",
    "L27": "https://aic-data.ledo.io.vn/Videos_L27_a.zip",
    "L28": "https://aic-data.ledo.io.vn/Videos_L28_a.zip",
    "L29": "https://aic-data.ledo.io.vn/Videos_L29_a.zip",
    "L30": "https://aic-data.ledo.io.vn/Videos_L30_a.zip",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, help="Range connections per video")
    parser.add_argument("--video-workers", type=int, default=2, help="Videos downloaded concurrently")
    parser.add_argument("--chunk-mb", type=int, default=4)
    args = parser.parse_args()
    values = json.loads(args.candidates.read_text(encoding="utf-8"))
    ids = values if isinstance(values, list) else values.get("video_ids", [])
    archives: dict[str, RangeZip] = {}
    jobs = []
    for video_id in sorted(set(ids)):
        prefix = video_id.split("_", 1)[0]
        if prefix not in ARCHIVES:
            raise ValueError(f"no archive mapping for {video_id}")
        archives.setdefault(prefix, RangeZip(ARCHIVES[prefix]))
        jobs.append((video_id, prefix))

    def download(job):
        video_id, prefix = job
        output = args.output / f"{video_id}.mp4"
        if not output.exists():
            last_error = None
            for attempt in range(3):
                try:
                    archives[prefix].download_parallel(
                        video_id, output, workers=args.workers,
                        chunk_size=args.chunk_mb * 1024 * 1024,
                    )
                    break
                except KeyError:
                    print(f"SKIP: {video_id} not found in archive")
                    return video_id, 0
                except Exception as error:
                    last_error = error
                    output.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"failed to download {video_id}") from last_error
        return video_id, output.stat().st_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.video_workers, len(jobs) or 1)) as pool:
        for video_id, size in pool.map(download, jobs):
            print(video_id, size)


if __name__ == "__main__":
    main()
