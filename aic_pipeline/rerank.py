"""Local cross-encoder reranking and selective remote adjudication."""

from __future__ import annotations

import json
import os
from typing import Any

import requests


def local_rerank(query: str, candidates: list[dict], limit: int = 100, model_name: str = "BAAI/bge-reranker-v2-m3") -> list[dict]:
    if not candidates:
        return []
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name)
    except Exception:
        return candidates[:limit]
    selected = candidates[:limit]
    pairs = [(query, item.get("text") or item.get("caption") or item.get("ocr") or "") for item in selected]
    try: scores = model.predict(pairs)
    except Exception: return candidates[:limit]
    for item, score in zip(selected, scores):
        item["rerank_score"] = float(score)
    selected.sort(key=lambda item: item["rerank_score"], reverse=True)
    return selected + candidates[limit:]


def should_judge(candidates: list[dict], always: bool = False) -> bool:
    if always:
        return bool(candidates)
    if len(candidates) < 2:
        return False
    first = float(candidates[0].get("rerank_score", candidates[0].get("retrieval_score", 0.0)))
    second = float(candidates[1].get("rerank_score", candidates[1].get("retrieval_score", 0.0)))
    return abs(first - second) < max(0.05, abs(first) * 0.08)


def remote_judge(query: str, candidates: list[dict], model: str | None = None) -> list[dict]:
    api_key = os.getenv("AIC_JUDGE_API_KEY", "")
    base_url = os.getenv("AIC_JUDGE_BASE_URL", "https://api.pateway.ai/v1").rstrip("/")
    model = model or os.getenv("AIC_JUDGE_MODEL", "gpt-5.6-terra")
    if not api_key or not candidates:
        return candidates
    compact = [{
        "rank": index + 1,
        "keyframe_id": item["keyframe_id"],
        "video_id": item["video_id"],
        "frame_id": item["frame_id"],
        "timestamp": item.get("timestamp"),
        "caption": item.get("caption", "")[:700],
        "ocr": item.get("ocr", "")[:300],
        "objects": item.get("objects", "")[:300],
    } for index, item in enumerate(candidates[:10])]
    prompt = (
        "Select the candidate frame that best answers the video retrieval query. "
        "Return strict JSON: {\"selected_rank\": integer, \"confidence\": number, \"reason\": string}.\n"
        f"Query: {query}\nCandidates: {json.dumps(compact, ensure_ascii=False)}"
    )
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            timeout=120,
        )
        response.raise_for_status()
        content: Any = response.json()["choices"][0]["message"]["content"]
    except (OSError, ValueError, KeyError, requests.RequestException):
        return candidates
    if isinstance(content, str):
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        decision = json.loads(content)
    else:
        decision = content
    selected = int(decision["selected_rank"]) - 1
    if selected < 0 or selected >= min(10, len(candidates)):
        return candidates
    chosen = candidates.pop(selected)
    chosen["judge_confidence"] = float(decision.get("confidence", 0.0))
    chosen["judge_reason"] = str(decision.get("reason", ""))
    candidates.insert(0, chosen)
    return candidates
