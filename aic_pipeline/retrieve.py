"""Lexical retrieval, RRF-compatible fusion, neighbors, and output export."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from .query import clip_english_query, expand_query
from .temporal import event_terms, suppress_duplicates, temporal_score
from .context import search_video_context

_dense_model = None
_dense_index = None
_dense_index_dir = None
_clip_model = None
_clip_processor = None
_clip_vectors = None
_clip_ids = None
_clip_feature_dir = None

# These words are useful to humans but occur in almost every caption or OCR
# line. Removing them from the OR FTS query both speeds up batch runs and lets
# distinctive visual entities carry the score. Keep this deliberately small:
# if a query contains only common words we fall back to every token below.
_FTS_STOPWORDS = {
    "a", "an", "and", "are", "at", "by", "for", "from", "in", "is", "it", "of",
    "on", "the", "to", "with", "after", "before", "then", "that", "this", "there",
    "các", "cảnh", "có", "của", "đang", "được", "khi", "là", "lần", "một", "những",
    "người", "này", "phần", "sau", "sau đó", "sẽ", "trên", "trong", "và", "với",
}


def _rows_for_keyframe_ids(connection: sqlite3.Connection, keyframe_ids: list[str]) -> dict[str, dict]:
    """Fetch candidate rows in batches instead of one SQLite query per hit."""
    rows: dict[str, dict] = {}
    columns = None
    for start in range(0, len(keyframe_ids), 400):
        batch = keyframe_ids[start:start + 400]
        if not batch:
            continue
        placeholders = ",".join("?" for _ in batch)
        cursor = connection.execute(
            f"SELECT * FROM keyframes WHERE keyframe_id IN ({placeholders})", batch
        )
        columns = [column[0] for column in cursor.description]
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            rows[item["keyframe_id"]] = item
    return rows


def _official_clip_keyframe_id(video_id: str, archive_number: int) -> str:
    """Convert the archive's zero-based position to the DB's one-based key."""
    return f"{video_id}:{int(archive_number) + 1}"


def _fts_query(query: str) -> str:
    raw_tokens = re.findall(r"[\wÀ-ỹ]+", query.lower())
    tokens = []
    for token in raw_tokens:
        if token not in _FTS_STOPWORDS and token not in tokens:
            tokens.append(token)
    if not tokens:
        tokens = raw_tokens
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def _sequence_fts_query(query: str) -> str:
    """Require evidence for each temporal clause in a video-context document.

    This is purposely used only for the per-video timeline: insisting that one
    *frame* mention every event would be incorrect. If a captioning gap makes
    the strict expression empty, callers retain the broad OR search as a
    fallback.
    """
    clauses = event_terms(query)
    if len(clauses) < 2:
        return ""
    groups = []
    for clause in clauses:
        english = clip_english_query(clause)
        expression = _fts_query(english or clause)
        if expression:
            groups.append(f"({expression})")
    return " AND ".join(groups) if len(groups) >= 2 else ""


def search_fts(connection: sqlite3.Connection, query: str, limit: int = 300) -> list[dict]:
    expression = _fts_query(query)
    if not expression:
        return []
    try:
        cursor = connection.execute(
        """SELECT k.*, bm25(keyframes_fts) AS bm25_score
           FROM keyframes_fts f JOIN keyframes k ON k.rowid=f.rowid
           WHERE keyframes_fts MATCH ? ORDER BY bm25_score LIMIT ?""",
            (expression, limit),
        )
    except sqlite3.OperationalError:
        return []
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def search_fts_in_videos(connection: sqlite3.Connection, query: str, video_ids: list[str],
                         limit: int = 300, per_video: int = 8) -> list[dict]:
    """Find frame evidence inside videos selected by the timeline sidecar.

    A video-level match alone is not submittable. This constrained FTS pass
    injects concrete official frames from the selected videos back into the
    normal rank fusion while preventing one long video from taking every slot.
    """
    expression = _fts_query(query)
    if not expression or not video_ids:
        return []
    rows: list[dict] = []
    for start in range(0, len(video_ids), 300):
        video_batch = video_ids[start:start + 300]
        placeholders = ",".join("?" for _ in video_batch)
        try:
            cursor = connection.execute(
                f"""SELECT k.*, bm25(keyframes_fts) AS bm25_score
                   FROM keyframes_fts f JOIN keyframes k ON k.rowid=f.rowid
                   WHERE keyframes_fts MATCH ? AND k.video_id IN ({placeholders})
                   ORDER BY bm25_score LIMIT ?""",
                [expression, *video_batch, max(limit * 3, 100)],
            )
        except sqlite3.OperationalError:
            continue
        columns = [column[0] for column in cursor.description]
        rows.extend(dict(zip(columns, row)) for row in cursor.fetchall())
    selected: list[dict] = []
    per_video_count: dict[str, int] = defaultdict(int)
    for row in sorted(rows, key=lambda item: float(item.get("bm25_score", 0.0))):
        video_id = row["video_id"]
        if per_video_count[video_id] >= per_video:
            continue
        selected.append(row)
        per_video_count[video_id] += 1
        if len(selected) >= limit:
            break
    return selected


def rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, item_id in enumerate(ranked, 1):
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


def neighbor_rows(connection: sqlite3.Connection, video_id: str, number: int, radius: int = 2) -> list[dict]:
    cursor = connection.execute(
        "SELECT * FROM keyframes WHERE video_id=? AND keyframe_number BETWEEN ? AND ? ORDER BY keyframe_number",
        (video_id, number - radius, number + radius),
    )
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def _dense_results(connection: sqlite3.Connection, query: str, dense_dir: Path | None, candidate_limit: int) -> list[dict]:
    if dense_dir is None or not (dense_dir / "meta.json").exists():
        return []
    from .dense import DenseIndex, TfidfIndex
    global _dense_index, _dense_index_dir

    meta = __import__("json").loads((dense_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("kind") == "tfidf":
        if _dense_index_dir != dense_dir or _dense_index is None:
            _dense_index = TfidfIndex(dense_dir)
            _dense_index_dir = dense_dir
        ranked = _dense_index.search(query, candidate_limit)
    elif (dense_dir / "vectors.npy").exists():
        ranked = None
    else:
        return []

    global _dense_model
    if ranked is None and _dense_model is None:
        from sentence_transformers import SentenceTransformer
        _dense_model = SentenceTransformer(
            __import__("os").getenv("AIC_EMBED_MODEL", "BAAI/bge-m3"),
            device=__import__("os").getenv("AIC_EMBED_DEVICE") or None,
        )
    if ranked is None:
        vector = _dense_model.encode(query, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        if _dense_index_dir != dense_dir or _dense_index is None:
            _dense_index = DenseIndex(dense_dir)
            _dense_index_dir = dense_dir
        ranked = _dense_index.search(vector, candidate_limit)
    row_map = _rows_for_keyframe_ids(connection, [keyframe_id for keyframe_id, _ in ranked])
    results = []
    for keyframe_id, score in ranked:
        item = row_map.get(keyframe_id)
        if item is None:
            continue
        item["dense_score"] = score
        results.append(item)
    return results

_feature_model = None
_feature_tokenizer = None


def _feature_results(connection: sqlite3.Connection, query: str, feature_dir: Path | None, limit: int) -> list[dict]:
    """Use precomputed CLIP/SigLIP vectors when BTC or a local extractor supplied them."""
    if feature_dir is None or not (feature_dir / "vectors.npy").exists(): return []
    global _feature_model, _feature_tokenizer
    model_name = __import__("os").getenv("AIC_CLIP_TEXT_MODEL", "openai/clip-vit-base-patch32")
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        if _feature_model is None or _feature_tokenizer is None:
            _feature_model = AutoModel.from_pretrained(model_name)
            _feature_tokenizer = AutoTokenizer.from_pretrained(model_name)
        with torch.no_grad():
            vector = _feature_model.get_text_features(**_feature_tokenizer(query, return_tensors="pt"))[0].numpy()
        vector = vector / max(np.linalg.norm(vector), 1e-8)
    except Exception:
        return []
    from .dense import DenseIndex
    ranked = DenseIndex(feature_dir).search(vector.astype("float32"), limit)
    row_map = _rows_for_keyframe_ids(connection, [keyframe_id for keyframe_id, _ in ranked])
    result = []
    for keyframe_id, score in ranked:
        item = row_map.get(keyframe_id)
        if item:
            item["clip_score"] = score
            result.append(item)
    return result


def _official_clip_results(connection: sqlite3.Connection, query: str, feature_dir: Path | None, limit: int) -> list[dict]:
    """Search the official per-video CLIP ViT-B/32 feature archive."""
    if feature_dir is None or not feature_dir.exists():
        return []
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch
    except Exception:
        return []
    global _clip_model, _clip_processor
    try:
        if _clip_model is None:
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = _clip_processor.tokenizer(query, return_tensors="pt", padding=True,
                                           truncation=True, max_length=77)
        with torch.no_grad():
            out = _clip_model.get_text_features(**inputs)
        # HF now returns BaseModelOutputWithPooling (pooler_output) instead of Tensor
        import torch as _torch
        if isinstance(out, _torch.Tensor):
            vector = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            vector = out.pooler_output
        elif hasattr(out, "text_embeds") and out.text_embeds is not None:  # older SigLIP-style
            vector = out.text_embeds  # type: ignore
        else:
            vector = out.last_hidden_state[:, 0, :]  # fallback
        vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        query_vector = vector[0].cpu().numpy().astype("float32")
    except Exception:
        return []
    global _clip_vectors, _clip_ids, _clip_feature_dir
    combined_vectors = feature_dir / "vectors.npy"
    combined_ids = feature_dir / "ids.json"
    if combined_vectors.exists() and combined_ids.exists():
        try:
            if _clip_feature_dir != feature_dir or _clip_vectors is None or _clip_ids is None:
                _clip_vectors = np.asarray(np.load(combined_vectors, mmap_mode="r"), dtype="float32")
                _clip_ids = __import__("json").loads(combined_ids.read_text(encoding="utf-8"))
                _clip_feature_dir = feature_dir
            vectors = _clip_vectors
            ids = _clip_ids
            scores = vectors @ query_vector
            chosen = np.argpartition(scores, -min(limit, len(scores)))[-min(limit, len(scores)):]
            chosen = chosen[np.argsort(scores[chosen])[::-1]]
            scored = [(float(scores[index]), *ids[int(index)].rsplit(":", 1)) for index in chosen]
        except (OSError, ValueError, KeyError):
            scored = []
    else:
        scored = []
    if not scored:
        scored = []
        # Fallback for an unbuilt archive index.
        for feature_path in sorted(feature_dir.glob("*.npy")):
            try:
                vectors = np.asarray(np.load(feature_path), dtype="float32")
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
                scores = vectors @ query_vector
                for index in np.argpartition(scores, -min(limit, len(scores)))[-min(limit, len(scores)):]:
                    scored.append((float(scores[index]), feature_path.stem, str(int(index))))
            except (OSError, ValueError):
                continue
        scored.sort(reverse=True)
    result = []
    # CLIP archive is 0-based (L21_V001:0) while DB keyframe_id is 1-based (L21_V001:1)
    key_ids = [_official_clip_keyframe_id(video_id, keyframe_number)
               for score, video_id, keyframe_number in scored[:limit]]
    row_map = _rows_for_keyframe_ids(connection, key_ids)
    for score, video_id, keyframe_number in scored[:limit]:
        keyframe_number = int(keyframe_number) + 1
        item = row_map.get(f"{video_id}:{keyframe_number}")
        if item is None:
            continue
        item["clip_score"] = score
        result.append(item)
    return result


def search(connection: sqlite3.Connection, query: str, limit: int = 20, candidate_limit: int = 300, neighbor_radius: int = 2, dense_dir: Path | None = None, feature_dir: Path | None = None, expansions: list[str] | None = None, preserve_video_coverage: bool = False, include_neighbors: bool = True) -> list[dict]:
    variants = expand_query(query, expansions)
    channels = []
    # The token-normalized third variant adds little semantic value but makes
    # the CPU TF-IDF fallback scan the whole 177k-row matrix again. Keep dense
    # retrieval to the original and translated variants; lexical retrieval can
    # still use every cheap variant.
    dense_variants = variants[:2]
    for index, variant in enumerate(variants):
        dense = _dense_results(connection, variant, dense_dir, candidate_limit) if index < len(dense_variants) else []
        channels.extend((search_fts(connection, variant, candidate_limit), dense))
    # Add a pure-English keyword search for FTS as well: the Vietnamese
    # variants contain many stopwords that drown the English caption terms.
    en_q = clip_english_query(query)
    if en_q and en_q not in variants:
        channels.extend((search_fts(connection, en_q, candidate_limit), []))
    lexical_lists = [channels[index] for index in range(0, len(channels), 2)]
    dense_lists = [channels[index] for index in range(1, len(channels), 2)]

    # Sequence-aware recovery. The context sidecar is optional; deployments
    # that have not built it retain the former frame-only behavior. Each video
    # context hit is converted back to actual frame candidates before fusion.
    # One original-language pass and one English-caption pass are enough. More
    # variants multiply constrained FTS work and made large query batches slow.
    context_variants = list(dict.fromkeys([variants[0], *( [en_q] if en_q else [])]))
    context_lists: list[list[dict]] = []
    strict_context = search_video_context(connection, _sequence_fts_query(query), limit=30)
    if strict_context:
        # The sequence match represents evidence from several cuts, so let it
        # outweigh a broad one-keyword video hit in video-level RRF.
        context_lists.extend((strict_context, strict_context, strict_context))
    for variant in context_variants:
        rows = search_video_context(connection, _fts_query(variant), limit=30)
        if rows:
            context_lists.append(rows)
    context_scores = rrf([[row["video_id"] for row in rows] for rows in context_lists])
    context_rank = {
        video_id: rank for rank, video_id in enumerate(
            sorted(context_scores, key=context_scores.get, reverse=True), 1
        )
    }
    if context_rank:
        selected_videos = list(context_rank)[:24]
        # A constrained pass per variant allows a video whose clues are spread
        # across cuts to supply candidate frames even if none was globally top.
        for variant in context_variants:
            scoped = search_fts_in_videos(connection, variant, selected_videos, min(80, candidate_limit))
            if scoped:
                lexical_lists.append(scoped)
    clip_lists = []
    if feature_dir is not None:
        if (feature_dir / "meta.json").exists():
            feature_rows = _feature_results(connection, query, feature_dir, candidate_limit)
            if feature_rows:
                clip_lists.append(feature_rows)
        if (feature_dir / "ids.json").exists() and (feature_dir / "vectors.npy").exists():
            # CLIP ViT-B/32 has weak Vietnamese support; use a pure-English
            # keyword query. Mixed Vi-En strings still contain Vietnamese
            # tokens that confuse the text encoder and return visually
            # irrelevant top hits.
            vq = clip_english_query(query)
            if vq:
                rows = _official_clip_results(connection, vq, feature_dir, candidate_limit)
                if rows:
                    clip_lists.append(rows)
    lexical = [row for ranked in lexical_lists for row in ranked]
    dense = [row for ranked in dense_lists for row in ranked]
    # Keep each retrieval channel independent. Flattening query variants into
    # one list lets duplicate hits from a single channel overpower the other
    # channels, especially for long Vietnamese queries.
    # Weight CLIP higher for KIS: visual evidence should dominate lexical
    # noise, especially when the query is Vietnamese but captions are English.
    lists = (
        [[row["keyframe_id"] for row in ranked] for ranked in lexical_lists]
        + [[row["keyframe_id"] for row in ranked] for ranked in dense_lists]
        + [[row["keyframe_id"] for row in ranked] for ranked in clip_lists] * 3
    )
    lists = [items for items in lists if items]
    fused = rrf(lists)
    clip = [row for ranked in clip_lists for row in ranked]
    by_id = {row["keyframe_id"]: row for row in lexical + dense + clip}
    ordered = sorted(fused, key=fused.get, reverse=True)
    results = []
    seen_videos: set[str] = set()
    for keyframe_id in ordered:
        row = by_id[keyframe_id]
        row["retrieval_score"] = fused[keyframe_id]
        text = (row.get("text") or "").lower()
        # Use English keywords for co-occurrence when the query is Vietnamese;
        # otherwise Vietnamese tokens never match English captions and penalize
        # the correct visual hits.
        co_query = clip_english_query(query) if any(ord(c) > 127 for c in query) else query
        query_tokens = set(re.findall(r"[\wÀ-ỹ]+", co_query.lower()))
        row["cooccurrence_score"] = sum(token in text for token in query_tokens) / max(1, len(query_tokens))
        row["temporal_score"] = temporal_score(row, query)
        row["retrieval_score"] += 0.02 * row["cooccurrence_score"] + 0.05 * row["temporal_score"]
        context_position = context_rank.get(row["video_id"])
        if context_position is not None:
            # This is intentionally smaller than a direct frame match: context
            # establishes that the video is plausible, while the frame-level
            # channels still decide the exact evidence position.
            row["retrieval_score"] += 0.012 / (context_position ** 0.5)
            row["video_context_rank"] = context_position
        # Direct clip similarity should influence final ranking for KIS
        if "clip_score" in row:
            row["retrieval_score"] += 0.1 * float(row["clip_score"])
        results.append(row)
        seen_videos.add(row["video_id"])
        # Gather a larger pool before deduplication. Otherwise a handful of
        # adjacent frames can consume the requested limit and leave a short,
        # low-coverage CSV after temporal suppression.
        if len(results) >= max(limit * 4, candidate_limit):
            break
    results.sort(key=lambda x: x["retrieval_score"], reverse=True)
    if preserve_video_coverage:
        # The official score uses R@1/R@5/R@20/R@50/R@100. Keep multiple
        # temporally separated frames from a relevant video so a correct
        # interval can still appear in a later cutoff.
        results = suppress_duplicates(results, per_video=max(10, limit // 2), min_gap=0.5)[:limit]
    else:
        results = suppress_duplicates(results, per_video=max(3, limit // 4))[:limit]
    if include_neighbors:
        for row in results:
            row["neighbors"] = neighbor_rows(connection, row["video_id"], row["keyframe_number"], neighbor_radius)
    return results


def export_submission(results: list[dict], output, limit: int = 100) -> None:
    import csv
    writer = csv.writer(output, lineterminator="\n")
    for row in results[:limit]: writer.writerow((row["video_id"], row["frame_id"]))
