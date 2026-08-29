"""Transcribe official Batch 1 MP4s with Deepgram, one resumable JSONL row/video."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import tempfile
import time
from pathlib import Path

import requests

from aic_pipeline.range_zip import RangeZip


BASE = "https://aic-data.ledo.io.vn/"
ARCHIVES = {
    **{f"L{i}": f"Videos_L{i}_a.zip" for i in [21, 22, 23, 24, 25, 27, 28, 29, 30]},
    **{f"L26_{part}": f"Videos_L26_{part}.zip" for part in "abcde"},
}
VIDEO_RE = re.compile(r"(L\d+_V\d+)", re.IGNORECASE)


def video_id(name: str) -> str | None:
    match = VIDEO_RE.search(Path(name).stem)
    return match.group(1).upper() if match else None


def done_rows(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" and row.get("video_id"):
            done.add(str(row["video_id"]).upper())
    return done


def transcribe(path: Path, vid: str, member: str, model: str,
               language: str, key: str) -> dict:
    params = {
        "model": model,
        "language": language,
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
    }
    headers = {
        "Authorization": f"Token {key}",
        "Content-Type": "video/mp4",
    }
    last: Exception | None = None
    for attempt in range(6):
        try:
            with path.open("rb") as stream:
                response = requests.post(
                    "https://api.deepgram.com/v1/listen",
                    params=params,
                    headers=headers,
                    data=stream,
                    timeout=900,
                )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"Deepgram transient HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            payload = response.json()
            channel = (payload.get("results", {}).get("channels", [{}]) or [{}])[0]
            alternative = (channel.get("alternatives", [{}]) or [{}])[0]
            transcript = str(alternative.get("transcript", ""))
            words = alternative.get("words") or []
            utterances = []
            for item in payload.get("results", {}).get("utterances", []) or []:
                utterances.append({
                    "start_ms": round(float(item.get("start", 0)) * 1000),
                    "end_ms": round(float(item.get("end", 0)) * 1000),
                    "text": str(item.get("transcript", "")),
                    "confidence": item.get("confidence"),
                })
            return {
                "status": "ok",
                "video_id": vid,
                "source_member": member,
                "provider": "deepgram",
                "model": model,
                "language": language,
                "transcript": transcript,
                "words": [{
                    "start_ms": round(float(item.get("start", 0)) * 1000),
                    "end_ms": round(float(item.get("end", 0)) * 1000),
                    "word": str(item.get("word", "")),
                    "confidence": item.get("confidence"),
                } for item in words],
                "utterances": utterances,
                "transcript_confidence": alternative.get("confidence"),
            }
        except Exception as error:  # noqa: BLE001
            last = error
            if attempt < 5:
                time.sleep(min(120, 5 * (attempt + 1)))
    return {
        "status": "error",
        "video_id": vid,
        "source_member": member,
        "provider": "deepgram",
        "model": model,
        "language": language,
        "error": repr(last),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=sorted(ARCHIVES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="nova-3")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--gcs-bucket", help="accepted for CLI compatibility; not used")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit("DEEPGRAM_API_KEY is missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = done_rows(args.output)
    archive = RangeZip(BASE + ARCHIVES[args.batch], block_size=4 * 1024 * 1024)
    members = []
    for name in archive.entries:
        if not name.lower().endswith(".mp4"):
            continue
        vid = video_id(name)
        if vid and vid not in done:
            members.append((vid, name))
    members.sort()
    if args.limit > 0:
        members = members[:args.limit]
    print(f"{args.batch}: {len(members)} video(s) cần ASR; đã có {len(done)}", flush=True)

    def run(item: tuple[str, str]) -> dict:
        vid, member = item
        with tempfile.TemporaryDirectory(prefix=f"deepgram-{vid}-") as temp:
            path = Path(temp) / f"{vid}.mp4"
            try:
                archive.download_parallel(member, path, workers=2,
                                          chunk_size=8 * 1024 * 1024)
                return transcribe(path, vid, member, args.model, args.language, key)
            except Exception as error:  # noqa: BLE001
                return {
                    "status": "error", "video_id": vid,
                    "source_member": member, "provider": "deepgram",
                    "model": args.model, "language": args.language,
                    "error": repr(error),
                }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        with args.output.open("a", encoding="utf-8") as handle:
            for result in pool.map(run, members):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{result['video_id']}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
