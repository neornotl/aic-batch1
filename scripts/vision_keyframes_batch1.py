"""Inspect official BTC keyframes with Gemini Vision and write RAG sidecars.

The archive is downloaded once by the workflow and is never extracted in full.
Each video is sent as ordered BTC keyframes (in bounded chunks) so the model can
use the real images while the output retains the official filename indices.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
import zipfile
from pathlib import Path


MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
VIDEO_RE = re.compile(r"(L\d+_V\d+)", re.IGNORECASE)

PROMPT = """Bạn là bộ lập chỉ mục hình ảnh cho hệ thống Video-RAG AIC2026.
Các ảnh đính kèm là TOÀN BỘ keyframe chính thức của MỘT video, theo đúng thứ tự.
Hãy quan sát ảnh thật, không suy đoán từ tên file và không bỏ qua các ảnh đã cung cấp.
Nhãn FRAME_INDEX là chỉ số tên JPG dùng để kiểm tra; giữ nguyên chính xác.

Trả về duy nhất JSON hợp lệ theo schema:
{
  "summary": "tóm tắt chi tiết diễn biến quan sát được theo thứ tự",
  "opening_scene": "cảnh đầu",
  "closing_scene": "cảnh cuối",
  "timeline": [{"keyframe_index": 123, "description": "..."}],
  "visual_entities": ["..."],
  "actions": ["..."],
  "on_screen_text": ["chữ đọc được"],
  "locations": ["..."],
  "search_keywords": ["từ khóa tiếng Việt và tiếng Anh"],
  "observed_keyframes": [{"keyframe_index": 123, "description": "mô tả ngắn đúng ảnh", "ocr": "", "entities": [], "actions": []}],
  "confidence": "high|medium|low"
}
Mỗi observed_keyframes phải tham chiếu các FRAME_INDEX trong phần ảnh hiện tại.
Nếu chữ hoặc chi tiết không đọc được, ghi chuỗi rỗng; không bịa. Đây là mô tả
thị giác để truy hồi, không phải OFFICIAL_FRAME_ID và không được tạo ID nộp bài."""


def _json_object(text: str) -> dict:
    value = (text or "").strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Gemini response did not contain a JSON object")


def _video_id(name: str) -> str | None:
    match = VIDEO_RE.search(name)
    return match.group(1).upper() if match else None


def _frame_index(name: str) -> int:
    return int(Path(name).stem)


def _done(path: Path) -> set[str]:
    result: set[str] = set()
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" and row.get("video_id"):
            result.add(str(row["video_id"]).upper())
    return result


def _id_list(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        values.update(item.strip().upper() for item in line.split(",") if item.strip())
    return values


def _usage(response: object) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    result = {}
    for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = int(value)
    return result


def inspect_video(archive_path: Path, video_id: str, members: list[str],
                  model: str, chunk_size: int) -> dict:
    api_key = os.environ.get("VERTEX_API_KEY")
    if not api_key:
        raise RuntimeError("VERTEX_API_KEY is missing")
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1"),
    )
    observations: list[dict] = []
    chunk_summaries: list[str] = []
    total_usage: dict[str, int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for start in range(0, len(members), chunk_size):
            current = members[start:start + chunk_size]
            contents: list[object] = [
                PROMPT,
                f"Video {video_id}. Chunk {start // chunk_size + 1}. "
                f"The following images are ordered keyframes {start + 1}-{start + len(current)}.",
            ]
            for member in current:
                contents.append(f"FRAME_INDEX={_frame_index(member)}")
                contents.append(types.Part.from_bytes(
                    data=archive.read(member), mime_type="image/jpeg"))
            last_error: Exception | None = None
            response = None
            for attempt in range(6):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0,
                            max_output_tokens=12000,
                        ),
                    )
                    break
                except Exception as error:  # noqa: BLE001
                    last_error = error
                    text = repr(error).upper()
                    transient = any(marker in text for marker in (
                        "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                        "DEADLINE_EXCEEDED", "INTERNAL",
                    ))
                    if attempt < 5:
                        time.sleep(min(120, (10 if transient else 5) * (attempt + 1)))
            if response is None:
                raise RuntimeError(f"chunk {start // chunk_size + 1} failed: {last_error!r}")
            parsed = _json_object(getattr(response, "text", "") or "")
            chunk_summaries.append(str(parsed.get("summary", "")).strip())
            chunk_obs = parsed.get("observed_keyframes", [])
            if isinstance(chunk_obs, list):
                observations.extend(item for item in chunk_obs if isinstance(item, dict))
            for key, value in _usage(response).items():
                total_usage[key] = total_usage.get(key, 0) + value
    return {
        "status": "ok",
        "video_id": video_id,
        "source_archive": archive_path.name,
        "model": model,
        "frame_count": len(members),
        "summary": " ".join(item for item in chunk_summaries if item),
        "opening_scene": observations[0].get("description", "") if observations else "",
        "closing_scene": observations[-1].get("description", "") if observations else "",
        "timeline": [{"keyframe_index": item.get("keyframe_index"),
                      "description": item.get("description", "")} for item in observations],
        "observed_keyframes": observations,
        "visual_entities": sorted({str(x) for item in observations
                                    for x in (item.get("entities") or [])}),
        "actions": sorted({str(x) for item in observations
                            for x in (item.get("actions") or [])}),
        "on_screen_text": sorted({str(item.get("ocr", "")) for item in observations
                                   if str(item.get("ocr", "")).strip()}),
        "locations": [],
        "search_keywords": [],
        "confidence": "high" if observations else "low",
        "usage_metadata": total_usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--model", default=os.environ.get("VERTEX_MODEL", MODEL))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-list", type=Path, default=None,
                        help="comma/newline-separated video IDs already covered elsewhere")
    args = parser.parse_args()
    if not args.archive.exists():
        raise SystemExit(f"archive not found: {args.archive}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _done(args.output) | _id_list(args.skip_list)
    grouped: dict[str, list[str]] = {}
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            video_id = _video_id(name)
            if video_id:
                grouped.setdefault(video_id, []).append(name)
    items = [(video_id, sorted(names, key=_frame_index))
             for video_id, names in sorted(grouped.items()) if video_id not in done]
    if args.limit > 0:
        items = items[:args.limit]
    print(f"{args.archive.name}: {len(items)} video(s) cần vision; đã có {len(done)}", flush=True)
    lock = threading.Lock()

    def run(item: tuple[str, list[str]]) -> dict:
        video_id, members = item
        try:
            return inspect_video(args.archive, video_id, members, args.model, args.chunk_size)
        except Exception as error:  # noqa: BLE001
            return {
                "status": "error",
                "video_id": video_id,
                "source_archive": args.archive.name,
                "model": args.model,
                "frame_count": len(members),
                "error": repr(error),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(run, items):
            with lock, args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{result['video_id']}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
