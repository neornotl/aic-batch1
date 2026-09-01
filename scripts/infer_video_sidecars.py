"""Fuse keyframe observations and Deepgram transcript with Gemini.

This is intentionally a text-sidecar pass: the expensive BTC keyframe vision
lane remains the source of visual observations, while this script lets Gemini
reason over those observations together with speech.  It is resumable and
records raw token usage for an auditable cost counter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path


PROMPT = """Bạn là bộ tổng hợp Video-RAG cho AIC2026.
Mỗi mục là dữ liệu của MỘT video: quan sát từ official BTC keyframes và
transcript Deepgram (nếu có). Hãy suy luận nội dung video CHỈ từ dữ liệu được
cung cấp; không bịa chi tiết không có evidence. Giữ đúng thứ tự nếu timeline
có mốc. Trả về DUY NHẤT JSON array, một object cho mỗi video_id:
[{"video_id":"...","summary":"tóm tắt diễn biến chi tiết nhưng ngắn gọn",
"visual_evidence":["..."],"speech_evidence":["..."],
"search_keywords":["..."],"confidence":"high|medium|low"}]
Nếu thiếu một modality, ghi rõ trong summary/evidence và hạ confidence.
"""


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(Path(p) for p in paths):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def by_video(rows: list[dict], *, visual: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        if row.get("status") not in (None, "ok"):
            continue
        video_id = str(row.get("video_id") or "").strip().upper()
        if not video_id:
            continue
        if visual:
            # Prefer a verified usable row. Keep a diagnostic placeholder for
            # a syntactically successful but empty upstream response.
            summary = str(row.get("summary") or "").strip()
            usable = bool(summary and row.get("observed_keyframes"))
            candidate = {
                "summary": summary,
                "opening_scene": str(row.get("opening_scene") or ""),
                "closing_scene": str(row.get("closing_scene") or ""),
                "timeline": row.get("timeline") or [],
                "observed_keyframes": row.get("observed_keyframes") or [],
                "visual_entities": row.get("visual_entities") or [],
                "actions": row.get("actions") or [],
                "on_screen_text": row.get("on_screen_text") or [],
                "usable": usable,
                "source_archive": row.get("source_archive"),
            }
            previous = out.get(video_id)
            if previous is None or (not previous.get("usable") and usable):
                out[video_id] = candidate
        else:
            transcript = str(row.get("transcript") or "").strip()
            if transcript:
                out[video_id] = {
                    "transcript": transcript,
                    "utterances": row.get("utterances") or [],
                    "language": row.get("language"),
                    "confidence": row.get("transcript_confidence"),
                    "source_member": row.get("source_member"),
                }
    return out


def usage(response: object) -> dict[str, int]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
        value = getattr(meta, key, None)
        if value is not None:
            out[key] = int(value)
    return out


def parse_array(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    value = (text or "").strip()
    for start, char in enumerate(value):
        if char != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
    raise ValueError("Gemini response did not contain a JSON array")


def parse_response(response: object) -> list[dict]:
    """Read structured output across google-genai SDK response variants."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    if isinstance(parsed, dict) and parsed.get("video_id"):
        return [parsed]
    text = str(getattr(response, "text", "") or "")
    if text.strip():
        return parse_array(text)
    # Some Vertex responses expose text only inside candidate parts.
    pieces: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            value = getattr(part, "text", None)
            if value:
                pieces.append(str(value))
    return parse_array("".join(pieces))


def compact_visual(value: dict) -> dict:
    observations = []
    for item in value.get("observed_keyframes", [])[:32]:
        if isinstance(item, dict):
            observations.append({
                "keyframe_index": item.get("keyframe_index"),
                "description": str(item.get("description") or "")[:300],
                "ocr": str(item.get("ocr") or "")[:160],
                "entities": (item.get("entities") or [])[:20],
                "actions": (item.get("actions") or [])[:20],
            })
    timeline = []
    for item in value.get("timeline", [])[:24]:
        if isinstance(item, dict):
            timeline.append({"keyframe_index": item.get("keyframe_index"),
                             "description": str(item.get("description") or "")[:300]})
    return {
        "summary": value.get("summary", "")[:5000],
        "opening_scene": value.get("opening_scene", "")[:500],
        "closing_scene": value.get("closing_scene", "")[:500],
        "timeline": timeline,
        "observed_keyframes": observations,
        "visual_entities": value.get("visual_entities", [])[:60],
        "actions": value.get("actions", [])[:60],
        "on_screen_text": value.get("on_screen_text", [])[:40],
        "usable": bool(value.get("usable")),
    }


def compact_transcript(value: dict) -> dict:
    utterances = []
    for item in value.get("utterances", [])[:120]:
        if isinstance(item, dict):
            utterances.append({"start_ms": item.get("start_ms"),
                               "end_ms": item.get("end_ms"),
                               "text": str(item.get("text") or "")[:500],
                               "confidence": item.get("confidence")})
    return {"transcript": value.get("transcript", "")[:12000],
            "utterances": utterances,
            "language": value.get("language"),
            "confidence": value.get("confidence")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--transcript-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-transcript-jsonl", type=Path, nargs="*", default=[])
    parser.add_argument("--only-new-transcripts", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    key = os.environ.get("VERTEX_API_KEY")
    if not key:
        raise SystemExit("VERTEX_API_KEY is missing")
    visual = by_video(read_jsonl(args.visual_jsonl), visual=True)
    transcripts = by_video(read_jsonl(args.transcript_jsonl), visual=False)
    baseline = by_video(read_jsonl(args.baseline_transcript_jsonl), visual=False)
    ids = sorted(set(visual) | set(transcripts))
    if args.only_new_transcripts:
        ids = [video_id for video_id in ids if video_id in transcripts and video_id not in baseline]
    done = set()
    if args.output.exists():
        for row in read_jsonl([args.output]):
            if row.get("status") == "ok" and row.get("video_id"):
                done.add(str(row["video_id"]).upper())
    ids = [video_id for video_id in ids if video_id not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cost_output.parent.mkdir(parents=True, exist_ok=True)
    import google.genai
    from google.genai import types
    client = google.genai.Client(vertexai=True, api_key=key,
                                 http_options=types.HttpOptions(api_version="v1"))
    chunks = [ids[start:start + max(1, args.batch_size)] for start in range(0, len(ids), max(1, args.batch_size))]
    lock = threading.Lock()
    totals = {"requests": 0, "ok": 0, "errors": 0,
              "prompt_token_count": 0, "candidates_token_count": 0,
              "total_token_count": 0}

    def run(chunk: list[str]) -> list[dict]:
        payload = []
        for video_id in chunk:
            payload.append({"video_id": video_id,
                            "keyframe_observations": compact_visual(visual.get(video_id, {})),
                            "deepgram": compact_transcript(transcripts[video_id]) if video_id in transcripts else None})
        last_error = None
        response = None
        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=[PROMPT, json.dumps(payload, ensure_ascii=False)],
                    config=types.GenerateContentConfig(response_mime_type="application/json",
                                                        temperature=0, max_output_tokens=6000),
                )
                parsed = parse_response(response)
                by_id = {str(item.get("video_id") or "").upper(): item for item in parsed}
                output = []
                for video_id in chunk:
                    item = by_id.get(video_id, {})
                    summary = str(item.get("summary") or "").strip()
                    if not summary:
                        raise ValueError(f"missing summary for {video_id}")
                    output.append({"status": "ok", "video_id": video_id,
                                   "summary": summary,
                                   "visual_evidence": item.get("visual_evidence") or [],
                                   "speech_evidence": item.get("speech_evidence") or [],
                                   "search_keywords": item.get("search_keywords") or [],
                                   "confidence": item.get("confidence") if item.get("confidence") in {"high", "medium", "low"} else "low",
                                   "model": args.model,
                                   "source": "keyframe_observations+deepgram",
                                   "pass": "new_transcript" if args.only_new_transcripts else "initial"})
                u = usage(response)
                with lock:
                    totals["requests"] += 1
                    totals["ok"] += len(output)
                    for name, value in u.items(): totals[name] += value
                return output
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                if attempt < 3:
                    time.sleep(5 * (attempt + 1))
        with lock:
            totals["requests"] += 1
        # A malformed response for a multi-video request should not poison the
        # entire batch. Retry at smaller granularity before recording errors.
        if len(chunk) > 1:
            midpoint = max(1, len(chunk) // 2)
            print(f"splitting failed chunk of {len(chunk)} videos", flush=True)
            return run(chunk[:midpoint]) + run(chunk[midpoint:])
        with lock:
            totals["errors"] += len(chunk)
        return [{"status": "error", "video_id": video_id, "error": last_error,
                 "model": args.model, "pass": "new_transcript" if args.only_new_transcripts else "initial"}
                for video_id in chunk]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(run, chunks))
    with args.output.open("a", encoding="utf-8") as handle:
        for batch in results:
            for row in batch:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    totals.update({"input_usd_per_million_tokens": float(os.environ.get("INPUT_USD_PER_MILLION", "0.30")),
                   "output_usd_per_million_tokens": float(os.environ.get("OUTPUT_USD_PER_MILLION", "2.50"))})
    totals["estimated_usd"] = (totals["prompt_token_count"] * totals["input_usd_per_million_tokens"]
                                + totals["candidates_token_count"] * totals["output_usd_per_million_tokens"]) / 1_000_000
    args.cost_output.write_text(json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **totals}, ensure_ascii=False), flush=True)
    return 0 if totals["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
