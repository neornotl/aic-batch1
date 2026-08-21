"""Make compact, one-frame-per-second reviews from small source-video windows.

The full MP4 exists only in a temporary directory.  Once frames have been
composited into a contact sheet, both the frame files and the MP4 are removed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from evidence_review.range_zip import RangeZip


ARCHIVES = {
    f"L{number}": f"https://aic-data.ledo.io.vn/Videos_L{number}_a.zip"
    for number in range(21, 31)
}
CELL_WIDTH, CELL_HEIGHT, HEADER_HEIGHT, COLUMNS = 320, 180, 30, 4


def download_video(video_id: str, output: Path) -> None:
    prefix = video_id.split("_", 1)[0]
    archive = RangeZip(ARCHIVES[prefix], block_size=4 * 1024 * 1024)
    archive.download(video_id, output)


def extract_window(video: Path, window: dict, frames: Path) -> list[Path]:
    start = int(window["start_frame"]) / float(window["fps"])
    duration = (int(window["end_frame"]) - int(window["start_frame"])) / float(window["fps"])
    pattern = str(frames / "%04d.jpg")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.6f}",
         "-i", str(video), "-t", f"{duration:.6f}",
         "-vf", f"fps={float(window.get('sample_fps', 1))}", "-q:v", "3", pattern],
        check=True,
    )
    return sorted(frames.glob("*.jpg"))


def render(window: dict, frame_paths: list[Path], output: Path) -> dict:
    rows = max(1, (len(frame_paths) + COLUMNS - 1) // COLUMNS)
    image = Image.new("RGB", (COLUMNS * CELL_WIDTH, rows * (CELL_HEIGHT + HEADER_HEIGHT)), "white")
    draw = ImageDraw.Draw(image)
    start = int(window["start_frame"])
    frame_step = float(window["fps"]) / float(window.get("sample_fps", 1))
    for index, path in enumerate(frame_paths):
        x, y = index % COLUMNS * CELL_WIDTH, index // COLUMNS * (CELL_HEIGHT + HEADER_HEIGHT)
        with Image.open(path) as source:
            thumb = ImageOps.fit(source.convert("RGB"), (CELL_WIDTH, CELL_HEIGHT), Image.Resampling.LANCZOS)
        image.paste(thumb, (x, y))
        approximate = round(start + index * frame_step)
        draw.rectangle((x, y + CELL_HEIGHT, x + CELL_WIDTH, y + CELL_HEIGHT + HEADER_HEIGHT), fill="#111827")
        draw.text((x + 6, y + CELL_HEIGHT + 8), f"~frame {approximate}", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=88, optimize=True)
    return {"id": window["id"], "video_id": window["video_id"], "sheet": output.name,
            "frames": len(frame_paths), "approximate": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = json.loads(args.spec.read_text(encoding="utf-8"))["windows"]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="aic-video-review-") as temporary:
        root = Path(temporary)
        by_video: dict[str, list[dict]] = {}
        for window in windows:
            by_video.setdefault(str(window["video_id"]), []).append(window)
        for video_id, video_windows in by_video.items():
            video = root / f"{video_id}.mp4"
            try:
                download_video(video_id, video)
                for window in video_windows:
                    frame_dir = root / window["id"]
                    try:
                        frame_dir.mkdir()
                        paths = extract_window(video, window, frame_dir)
                        results.append(render(window, paths, args.output / "sheets" / f"{window['id']}.jpg"))
                    finally:
                        shutil.rmtree(frame_dir, ignore_errors=True)
            finally:
                video.unlink(missing_ok=True)
    (args.output / "review.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"windows": len(results), "frames": sum(item["frames"] for item in results)}))


if __name__ == "__main__":
    main()
