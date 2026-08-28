"""Summarize every official BTC video in one logical batch.

The source archives are read through HTTP ranges, so the runner only keeps the
current video on disk. Results are appended as they finish and can be resumed
without reprocessing successful video IDs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

# This module is tracked in the repository.  Do not depend on the local-only
# evidence_review checkout, which is not present on GitHub Actions runners.
from aic_pipeline.range_zip import RangeZip


BASE = "https://aic-data.ledo.io.vn/"
ARCHIVES = {
    **{f"L{i}": f"Videos_L{i}_a.zip" for i in [21, 22, 23, 24, 25, 27, 28, 29, 30]},
    **{f"L26_{part}": f"Videos_L26_{part}.zip" for part in "abcde"},
}

PROMPT = """Bạn là chuyên gia lập chỉ mục video cho hệ thống Video-RAG.
Hãy xem toàn bộ video được cung cấp và trả về JSON hợp lệ, không markdown, theo schema:
{
  "summary": "Tóm tắt chi tiết nội dung và diễn biến chính theo đúng thứ tự thời gian",
  "opening_scene": "Cảnh mở đầu, bối cảnh, nhân vật và vật thể nổi bật",
  "timeline": [{"approx_time": "MM:SS", "description": "Sự kiện/cảnh quan trọng"}],
  "visual_entities": ["người, vật thể, địa điểm, thương hiệu, món ăn, thiết bị..."],
  "actions": ["các hành động chính"],
  "on_screen_text": "Toàn bộ chữ/tiêu đề/biển báo có thể đọc được",
  "locations": ["các địa điểm hoặc bối cảnh"],
  "closing_scene": "Cảnh kết thúc hoặc trạng thái cuối video",
  "search_keywords": ["từ khóa tiếng Việt và tiếng Anh giúp truy hồi"],
  "confidence": "high|medium|low"
}
Ưu tiên chi tiết có thể dùng để tìm video khi có câu hỏi: nhân vật, hành động,
đồ vật, món ăn, địa danh, chữ trên màn hình và thứ tự diễn biến. Nếu không thấy
thông tin nào, dùng chuỗi rỗng hoặc mảng rỗng; không suy đoán."""


REQUIRED_FIELDS = {
    "summary": str,
    "opening_scene": str,
    "timeline": list,
    "visual_entities": list,
    "actions": list,
    "on_screen_text": (str, list),
    "locations": list,
    "closing_scene": str,
    "search_keywords": list,
    "confidence": str,
}


class ResponseParseError(ValueError):
    """The model returned text that cannot be trusted as a summary object."""


def _json_candidates(value: str) -> list[str]:
    """Return plausible JSON fragments without trying to invent missing data."""
    candidates = [value]
    if "```" in value:
        parts = value.split("```")
        for index in range(1, len(parts), 2):
            fenced = parts[index].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].lstrip()
            candidates.append(fenced)

    # Models sometimes append a short explanation after a valid object.  The
    # decoder lets us keep the first complete object while rejecting malformed
    # JSON (e.g. unescaped quotes or missing commas) rather than guessing.
    decoder = json.JSONDecoder()
    for start, char in enumerate(value):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            continue
        candidates.append(value[start:start + end])
    return candidates


def _validate_summary(parsed: object) -> dict:
    if not isinstance(parsed, dict):
        raise ResponseParseError("top-level JSON is not an object")
    missing = [field for field in REQUIRED_FIELDS if field not in parsed]
    if missing:
        raise ResponseParseError("missing required field(s): " + ", ".join(missing))
    wrong_type = [
        field for field, expected in REQUIRED_FIELDS.items()
        if not isinstance(parsed[field], expected)
    ]
    if wrong_type:
        raise ResponseParseError("wrong type for field(s): " + ", ".join(wrong_type))
    if not str(parsed["summary"]).strip():
        raise ResponseParseError("summary is empty")
    if parsed["confidence"].lower() not in {"high", "medium", "low"}:
        raise ResponseParseError("confidence must be high, medium, or low")
    return parsed


def video_id_from_member(name: str) -> str | None:
    match = re.search(r"(L\d+_V\d+)", Path(name).stem, re.IGNORECASE)
    return match.group(1).upper() if match else None


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" and row.get("video_id"):
            done.add(str(row["video_id"]))
    return done


def parse_response(text: str) -> dict:
    value = (text or "").strip()
    if not value:
        raise ResponseParseError("empty model response")
    errors: list[str] = []
    seen: set[str] = set()
    for candidate in _json_candidates(value):
        candidate = candidate.strip()
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _validate_summary(json.loads(candidate))
        except (json.JSONDecodeError, ResponseParseError) as error:
            errors.append(str(error))
    detail = errors[-1] if errors else "no JSON object found"
    raise ResponseParseError(detail)


def is_transient_vertex_error(error: Exception) -> bool:
    """Return whether an error is worth retrying after a quota cooldown."""
    text = repr(error).upper()
    return any(marker in text for marker in (
        "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
        "DEADLINE_EXCEEDED", "INTERNAL",
    ))


def summarize_one(video_id: str, member: str, archive: RangeZip, out: Path,
                  model: str, batch: str, bucket_name: str) -> dict:
    """Download one source video, summarize it through Vertex, then clean up.

    The old implementation used ``client.files.upload``.  That endpoint belongs
    to the Gemini Developer API and rejects a Vertex-restricted key.  Vertex
    accepts a Cloud Storage URI instead, so the runner uploads the temporary
    source to the project bucket and gives Vertex only that URI.  The object is
    deleted in ``finally`` even when generation fails.
    """
    api_key = os.environ.get("VERTEX_API_KEY")
    if not api_key:
        raise RuntimeError("VERTEX_API_KEY is missing")
    from google import genai
    from google.genai import types
    from google.cloud import storage

    # Express mode keeps the existing Vertex API-key setup.  Cloud Storage is
    # authenticated separately through GOOGLE_APPLICATION_CREDENTIALS, which
    # is written by the workflow from the dedicated service-account secret.
    client = genai.Client(
        vertexai=True,
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1"),
    )
    storage_client = storage.Client(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT") or "aic2026-drive-ingest"
    )
    bucket = storage_client.bucket(bucket_name)
    blob_name = f"aic-batch1-summary/{batch}/{video_id}.mp4"
    blob = bucket.blob(blob_name)
    blob.chunk_size = 8 * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix=f"aic-summary-{video_id}-") as temp:
        video_path = Path(temp) / f"{video_id}.mp4"
        download_workers = max(1, min(
            4, int(os.environ.get("RANGE_DOWNLOAD_WORKERS", "2"))))
        archive.download_parallel(
            member,
            video_path,
            workers=download_workers,
            chunk_size=8 * 1024 * 1024,
        )
        try:
            blob.upload_from_filename(str(video_path), content_type="video/mp4", timeout=900)
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_uri(
                        file_uri=f"gs://{bucket_name}/{blob_name}",
                        mime_type="video/mp4",
                    ),
                    PROMPT,
                ],
            )
            raw_response = getattr(response, "text", "") or ""
            try:
                parsed = parse_response(raw_response)
            except ResponseParseError as error:
                # Keep malformed model output visible and retryable.  It must
                # not be labelled as a successful low-confidence summary.
                # The cap prevents a bad response from bloating JSONL files.
                cap = 12000
                return {
                    "status": "error",
                    "video_id": video_id,
                    "source_member": member,
                    "model": model,
                    "error": f"response_parse: {error}",
                    "parse_error": str(error),
                    "raw_response": raw_response[:cap],
                    "raw_response_truncated": len(raw_response) > cap,
                }
            return {"status": "ok", "video_id": video_id, "source_member": member,
                    "model": model, **parsed}
        finally:
            try:
                blob.delete(timeout=120)
            except Exception as cleanup_error:  # noqa: BLE001
                # Do not hide the generation error, but make leaked objects
                # visible in the runner log for later cleanup.
                print(f"{video_id}: GCS cleanup warning: {cleanup_error!r}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=sorted(ARCHIVES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--video-ids", default="",
                        help="Comma-separated IDs to process (empty = whole batch)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many videos (0 = whole batch)")
    parser.add_argument("--gcs-bucket",
                        default=os.environ.get("GCS_BUCKET",
                                               "run-sources-aic2026-drive-ingest-us-central1"))
    parser.add_argument("--model", default=os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    requested_ids = {
        item.strip().upper() for item in args.video_ids.split(",") if item.strip()
    }
    archive = RangeZip(BASE + ARCHIVES[args.batch], block_size=4 * 1024 * 1024)
    members = []
    for name in archive.entries:
        video_id = video_id_from_member(name)
        if (video_id and name.lower().endswith(".mp4") and video_id not in done
                and (not requested_ids or video_id in requested_ids)):
            members.append((video_id, name))
    members.sort()
    if args.limit > 0:
        members = members[:args.limit]
    print(f"{args.batch}: {len(members)} video(s) còn phải tóm tắt; đã có {len(done)}", flush=True)
    lock = threading.Lock()

    def run(item: tuple[str, str]) -> dict:
        video_id, member = item
        last = None
        for attempt in range(6):
            try:
                return summarize_one(video_id, member, archive, args.output,
                                     args.model, args.batch, args.gcs_bucket)
            except Exception as error:  # noqa: BLE001
                last = error
                if attempt < 5:
                    if is_transient_vertex_error(error):
                        # Vertex quota errors often need a real cooldown; the
                        # old 5/10-second retry loop exhausted too quickly.
                        time.sleep(min(120, 10 * (2 ** attempt)))
                    else:
                        time.sleep(5 * (attempt + 1))
        return {"status": "error", "video_id": video_id, "source_member": member,
                "error": repr(last)}

    ok_count = 0
    error_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(run, members):
            with lock, args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{result['video_id']}: {result['status']}", flush=True)
            if result["status"] == "ok":
                ok_count += 1
            else:
                error_count += 1

    print(f"{args.batch}: ok={ok_count} error={error_count}", flush=True)
    if members and ok_count == 0:
        raise SystemExit("No successful Vertex summaries were produced; refusing a green workflow")


if __name__ == "__main__":
    main()
