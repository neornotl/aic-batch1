"""Evaluate ranked keyframes against accepted video/frame ranges."""

from __future__ import annotations

import json
from pathlib import Path


def evaluate(predictions: Path, ground_truth: Path, cutoffs: tuple[int, ...] = (1, 5, 10, 20, 50, 100)) -> dict:
    truth = {item["query_id"]: item for item in _read_jsonl(ground_truth)}
    grouped: dict[str, list[dict]] = {}
    for item in _read_jsonl(predictions):
        grouped.setdefault(item["query_id"], []).append(item)
    hits = {cutoff: 0 for cutoff in cutoffs}
    reciprocal_rank = 0.0
    evaluated = 0
    for query_id, spec in truth.items():
        ranked = sorted(grouped.get(query_id, []), key=lambda item: int(item.get("rank", 10**9)))
        accepted = spec.get("accepted", [])
        first_hit = None
        for position, candidate in enumerate(ranked, 1):
            rank = int(candidate.get("rank", position))
            if any(
                candidate["video_id"] == target["video_id"]
                and int(target["min_frame"]) <= int(candidate["frame_id"]) <= int(target["max_frame"])
                for target in accepted
            ):
                first_hit = rank
                break
        evaluated += 1
        if first_hit:
            reciprocal_rank += 1.0 / first_hit
            for cutoff in cutoffs:
                hits[cutoff] += int(first_hit <= cutoff)
    denominator = max(1, evaluated)
    recalls = {f"recall@{cutoff}": hits[cutoff] / denominator for cutoff in cutoffs}
    return {"queries": evaluated, **recalls, "mrr": reciprocal_rank / denominator}


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
