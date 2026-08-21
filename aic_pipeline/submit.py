"""Official AIC query batch runner, CSV validation, and ZIP packaging."""

from __future__ import annotations

import csv
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from .retrieve import search, search_fts

try:
    from .terra import TerraAdapter
except ImportError:
    TerraAdapter = None  # type: ignore


def query_kind(path: Path) -> str:
    suffix = path.stem.lower().split("-")[-1]
    if suffix not in {"kis", "qa", "trake"}:
        raise ValueError(f"Cannot infer query type from {path.name}")
    return suffix


def read_query(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def _answer_local(row: dict, query: str) -> str:
    """Conservative answer fallback from OCR/caption, capped for BTC format."""
    ocr = " ".join(str(row.get("ocr", "")).split())
    if ocr:
        return ocr[:100]
    caption = " ".join(str(row.get("caption", "")).split())
    return caption[:100]


def _answer_terra(row: dict, query: str, adapter: TerraAdapter | None) -> str:
    """Use Terra to generate a concise answer from the top candidate."""
    if adapter is None or not adapter.key:
        return ""
    compact = {
        "query": query,
        "candidate": {
            "video_id": row["video_id"],
            "frame_id": row["frame_id"],
            "timestamp": row.get("timestamp"),
            "caption": row.get("caption", "")[:1000],
            "ocr": row.get("ocr", "")[:500],
            "objects": row.get("objects", "")[:500],
            "asr": row.get("asr", "")[:500],
        },
    }
    prompt = (
        "Answer the question based on the candidate frame information. "
        "Return strict JSON: {\"answer\": string, \"confidence\": number}.\n"
        f"Candidate: {json.dumps(compact, ensure_ascii=False)}"
    )
    result = adapter.complete(prompt)
    if isinstance(result, dict) and "answer" in result:
        answer = str(result["answer"]).strip()
        return answer[:100]
    return ""


def _rerank_terra(query: str, rows: list[dict], adapter: TerraAdapter | None) -> tuple[list[dict], str]:
    """Select the best candidate, and optionally answer a QA query, with Terra."""
    if adapter is None or not adapter.key or not rows:
        return rows, ""
    candidates = [{
        "rank": index + 1,
        "video_id": row["video_id"],
        "frame_id": row["frame_id"],
        "timestamp": row.get("timestamp"),
        "caption": row.get("caption", "")[:700],
        "ocr": row.get("ocr", "")[:300],
        "objects": row.get("objects", "")[:300],
    } for index, row in enumerate(rows[:10])]
    prompt = (
        "Choose the candidate frame that best matches the query. "
        "Return strict JSON with selected_rank and answer. "
        "For non-QA retrieval, answer may be an empty string. "
        "Format: {\"selected_rank\": integer, \"answer\": string}.\n"
        f"Query: {query}\nCandidates: {json.dumps(candidates, ensure_ascii=False)}"
    )
    result = adapter.complete(prompt)
    try:
        selected = int(result.get("selected_rank", 1)) - 1
    except (AttributeError, TypeError, ValueError):
        return rows, ""
    if selected < 0 or selected >= min(10, len(rows)):
        return rows, ""
    chosen = rows[selected]
    ordered = [chosen] + [row for index, row in enumerate(rows) if index != selected]
    answer = str(result.get("answer", "") or "").strip()[:100]
    return ordered, answer


def _event_queries(query: str) -> list[str]:
    # First try to extract E1, E2, E3... patterns
    e_pattern = r'E\d+\s*[:：]\s*(.*?)(?=E\d+\s*[:：]|$)'
    matches = re.findall(e_pattern, query, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    # Fallback to keyword-based split
    parts = [p.strip() for p in re.split(
        r"\b(?:then|and then|after that|sau đó|sau khi|trước khi|rồi|before|after)\b",
        query,
        flags=re.IGNORECASE,
    ) if p.strip() and len(p.strip()) > 10]
    return parts or [query]


def _trake_terra(events: list[str], ranked: list[list[dict]], adapter: TerraAdapter | None) -> list[list[str]]:
    """Use Terra for temporal reasoning across events."""
    if adapter is None or not adapter.key:
        return []
    if not ranked or any(not group for group in ranked):
        return []
    videos = set(row["video_id"] for row in ranked[0])
    for group in ranked[1:]:
        videos &= {row["video_id"] for row in group}
    if not videos:
        return []

    # Keep remote reasoning bounded: the local ranking has already produced
    # the strongest shared-video candidates, so probing every video is wasteful.
    ranked_videos = sorted(videos, key=lambda video: sum(
        max(float(row.get("retrieval_score", 0.0)) for row in group if row["video_id"] == video)
        for group in ranked
    ), reverse=True)[:3]
    for video in ranked_videos:
        paths = [[row for row in group if row["video_id"] == video] for group in ranked]
        compact = {
            "video_id": video,
            "events": [{"event": e, "candidates": [
                {"frame_id": row["frame_id"], "timestamp": row.get("timestamp"),
                 "caption": row.get("caption", "")[:500], "objects": row.get("objects", "")[:300]}
                for row in group[:5]
            ]} for e, group in zip(events, paths) if group],
        }
        prompt = (
            "Select the best frame sequence for each event in temporal order. "
            "Return strict JSON: {\"sequence\": [{\"event_index\": int, \"frame_id\": int, \"confidence\": number}]}\n"
            f"Data: {json.dumps(compact, ensure_ascii=False)}"
        )
        result = adapter.complete(prompt)
        if isinstance(result, dict) and "sequence" in result:
            frames = [str(item["frame_id"]) for item in sorted(result["sequence"], key=lambda x: x["event_index"])]
            if len(frames) == len(events):
                return [[video, *frames]]
    return []


def run_query_file(path: Path, connection, dense_dir: Path | None = None, limit: int = 100,
                   terra_adapter: Any | None = None, feature_dir: Path | None = None,
                   candidate_limit: int = 300) -> list[list[str]]:
    kind = query_kind(path)
    query = read_query(path)
    if kind == "kis":
        # Keep the proven lexical ranking stable. Caption-only remote reranking
        # can move visually correct frames down the list.
        rows = search(connection, query, limit=limit, candidate_limit=candidate_limit, dense_dir=dense_dir, feature_dir=feature_dir)
        return [[row["video_id"], str(row["frame_id"])] for row in rows[:limit]]
    if kind == "qa":
        rows = search(connection, query, limit=limit, candidate_limit=candidate_limit, dense_dir=dense_dir, feature_dir=feature_dir)
        if not rows:
            return []
        results: list[list[str]] = []
        for row in rows[:limit]:
            answer = ""
            if terra_adapter and terra_adapter.key:
                # Terra may improve the answer text, but must not change the
                # locally retrieved frame ranking without visual evidence.
                answer = _answer_terra(row, query, terra_adapter)
            if not answer:
                answer = _answer_local(row, query)
            results.append([row["video_id"], str(row["frame_id"]), answer])
        return results
    events = _event_queries(query)
    # Use FTS-only with small candidate limit for TRAKE speed (TF-IDF is too slow for 177k docs)
    ranked = []
    for event in events:
        group = search_fts(connection, event, max(1000, candidate_limit * 3))
        for rank, row in enumerate(group, 1):
            row["retrieval_score"] = 1.0 / rank
        ranked.append(group)
    if not ranked or any(not group for group in ranked):
        return []
    # Rank videos by event coverage before temporal alignment. Requiring the
    # strict intersection of small per-event result sets drops the correct
    # video when one event has weak or missing caption/OCR evidence.
    video_scores: dict[str, float] = {}
    video_coverage: dict[str, int] = {}
    for group in ranked:
        best_by_video: dict[str, float] = {}
        for row in group:
            video = row["video_id"]
            best_by_video[video] = max(best_by_video.get(video, float("-inf")),
                                      float(row.get("retrieval_score", 0.0)))
        for video, score in best_by_video.items():
            video_scores[video] = video_scores.get(video, 0.0) + score
            video_coverage[video] = video_coverage.get(video, 0) + 1
    candidate_videos = [video for video in video_scores
                        if video_coverage[video] == len(events)]
    if not candidate_videos:
        return []
    candidate_videos.sort(key=lambda video: video_scores[video], reverse=True)
    ranked = [[row for row in group if row["video_id"] in candidate_videos]
              for group in ranked]
    # Try Terra for TRAKE first
    if terra_adapter and terra_adapter.key:
        terra_results = _trake_terra(events, ranked, terra_adapter)
        if terra_results:
            return terra_results[:limit]
    # Fallback to local DP
    videos = set(row["video_id"] for row in ranked[0])
    for group in ranked[1:]:
        videos &= {row["video_id"] for row in group}
    rows: list[list[str]] = []
    for video in videos:
        paths = [[row for row in group if row["video_id"] == video] for group in ranked]
        states: list[list[tuple[float, list[dict]]]] = []
        for event_index, candidates in enumerate(paths):
            current = []
            for row in candidates:
                score = float(row.get("retrieval_score", 0.0))
                timestamp = row.get("timestamp") or 0.0
                best = (score, [row])
                if event_index:
                    previous = [state for state in states[-1]
                                if (state[1][-1].get("timestamp") or 0.0) < timestamp]
                    if previous:
                        prior = max(previous, key=lambda state: state[0])
                        best = (prior[0] + score, prior[1] + [row])
                current.append(best)
            states.append(current)
        if states and states[-1]:
            best = max(states[-1], key=lambda state: state[0])
            if len(best[1]) == len(events):
                rows.append([video, *[str(row["frame_id"]) for row in best[1]]])
    return rows[:limit]


def write_csv(rows: list[list[str]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows[:100])
    return min(len(rows), 100)


def validate_csv(path: Path, kind: str) -> list[str]:
    errors: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return ["file is empty"]
    if len(rows) > 100:
        errors.append("more than 100 rows")
    expected_extra = {"kis": 1, "qa": 2, "trake": None}[kind]
    for line_no, row in enumerate(rows, 1):
        if len(row) < 1 + (expected_extra or 0):
            errors.append(f"line {line_no}: too few columns")
            continue
        if not re.fullmatch(r"L\d{2}_V\d{3}", row[0].strip()):
            errors.append(f"line {line_no}: invalid video id")
        if kind != "qa" or len(row) >= 2:
            frame_start = 1
            frame_end = 2 if kind in {"kis", "qa"} else len(row)
            for value in row[frame_start:frame_end]:
                if not re.fullmatch(r"\d+", value.strip()):
                    errors.append(f"line {line_no}: invalid frame id")
        if kind == "qa" and len(row[2]) > 100:
            errors.append(f"line {line_no}: answer exceeds 100 characters")
    return errors


def package_submission(csv_dir: Path, output_zip: Path) -> dict:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise ValueError("no CSV files found")
    errors = []
    for path in files:
        try:
            kind = query_kind(path)
            errors.extend(f"{path.name}: {error}" for error in validate_csv(path, kind))
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        raise ValueError("; ".join(errors))
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"submission/{path.name}")
    return {"files": len(files), "zip": str(output_zip), "errors": []}
