from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def transcribe(path: Path) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    try:
        model = WhisperModel(os.getenv("AIC_WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(path), language=os.getenv("AIC_WHISPER_LANGUAGE", "vi"))
        return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]
    except Exception as e:
        print(f"ASR failed for {path}: {e}")
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "evidence.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for video in sorted(args.videos.glob("*.mp4")):
            cache = args.output / f"{video.stem}.json"
            if cache.exists():
                item = json.loads(cache.read_text(encoding="utf-8"))
            else:
                item = {"video_id": video.stem, "path": str(video), "asr": transcribe(video)}
                cache.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(video.name, len(item.get("asr", [])), "segments")


if __name__ == "__main__":
    main()
