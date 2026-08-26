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
    if value.startswith("```"):
        value = value.split("```", 2)[1].removeprefix("json").strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"summary": value[:12000], "confidence": "low"}
    return parsed if isinstance(parsed, dict) else {"summary": value[:12000], "confidence": "low"}


def summarize_one(video_id: str, member: str, archive: RangeZip, out: Path, model: str) -> dict:
    api_key = os.environ.get("VERTEX_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("VERTEX_API_KEY/GEMINI_API_KEY is missing")
    from google import genai

    client = genai.Client(api_key=api_key)
    with tempfile.TemporaryDirectory(prefix=f"aic-summary-{video_id}-") as temp:
        video_path = Path(temp) / f"{video_id}.mp4"
        archive.download(member, video_path)
        uploaded = None
        try:
            # Files API avoids putting a potentially large source video in the
            # request body. The temporary Gemini file is deleted after use.
            uploaded = client.files.upload(file=str(video_path))
            for _ in range(120):
                state = getattr(getattr(uploaded, "state", None), "name", "")
                if state != "PROCESSING":
                    break
                time.sleep(5)
                uploaded = client.files.get(name=uploaded.name)
            response = client.models.generate_content(model=model, contents=[uploaded, PROMPT])
            parsed = parse_response(getattr(response, "text", ""))
            return {"status": "ok", "video_id": video_id, "source_member": member,
                    "model": model, **parsed}
        finally:
            if uploaded is not None:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=sorted(ARCHIVES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default=os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    archive = RangeZip(BASE + ARCHIVES[args.batch], block_size=4 * 1024 * 1024)
    members = []
    for name in archive.entries:
        video_id = video_id_from_member(name)
        if video_id and name.lower().endswith(".mp4") and video_id not in done:
            members.append((video_id, name))
    members.sort()
    print(f"{args.batch}: {len(members)} video(s) còn phải tóm tắt; đã có {len(done)}", flush=True)
    lock = threading.Lock()

    def run(item: tuple[str, str]) -> dict:
        video_id, member = item
        last = None
        for attempt in range(3):
            try:
                return summarize_one(video_id, member, archive, args.output, args.model)
            except Exception as error:  # noqa: BLE001
                last = error
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        return {"status": "error", "video_id": video_id, "source_member": member,
                "error": repr(last)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(run, members):
            with lock, args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{result['video_id']}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
