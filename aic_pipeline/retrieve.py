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
from .query import expand_query
from .temporal import suppress_duplicates, temporal_score

_dense_model = None
_dense_index = None
_dense_index_dir = None
_clip_model = None
_clip_processor = None
_clip_vectors = None
_clip_ids = None
_clip_feature_dir = None


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


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[\wÀ-ỹ]+", query.lower())
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


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
    results = []
    for keyframe_id, score in ranked:
        cursor = connection.execute("SELECT * FROM keyframes WHERE keyframe_id=?", (keyframe_id,))
        row = cursor.fetchone()
        if row is None:
            continue
        columns = [column[0] for column in cursor.description]
        item = dict(zip(columns, row))
        item["dense_score"] = score
        results.append(item)
    return results

def _feature_results(connection: sqlite3.Connection, query: str, feature_dir: Path | None, limit: int) -> list[dict]:
    """Use precomputed CLIP/SigLIP vectors when BTC or a local extractor supplied them."""
    if feature_dir is None or not (feature_dir / "vectors.npy").exists(): return []
    model_name = __import__("os").getenv("AIC_CLIP_TEXT_MODEL", "openai/clip-vit-base-patch32")
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        model, tokenizer = AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
        with torch.no_grad():
            vector = model.get_text_features(**tokenizer(query, return_tensors="pt"))[0].numpy()
        vector = vector / max(np.linalg.norm(vector), 1e-8)
    except Exception:
        return []
    from .dense import DenseIndex
    result=[]
    for keyframe_id, score in DenseIndex(feature_dir).search(vector.astype("float32"), limit):
        cur=connection.execute("SELECT * FROM keyframes WHERE keyframe_id=?", (keyframe_id,)); row=cur.fetchone()
        if row:
            item=dict(zip([c[0] for c in cur.description], row)); item["clip_score"]=score; result.append(item)
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
            vector = _clip_model.get_text_features(**inputs)
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
    key_ids = [f"{video_id}:{int(keyframe_number)}" for score, video_id, keyframe_number in scored[:limit]]
    row_map = _rows_for_keyframe_ids(connection, key_ids)
    for score, video_id, keyframe_number in scored[:limit]:
        keyframe_number = int(keyframe_number)
        item = row_map.get(f"{video_id}:{keyframe_number}")
        if item is None:
            continue
        item["clip_score"] = score
        result.append(item)
    return result


def search(connection: sqlite3.Connection, query: str, limit: int = 20, candidate_limit: int = 300, neighbor_radius: int = 2, dense_dir: Path | None = None, feature_dir: Path | None = None, expansions: list[str] | None = None, preserve_video_coverage: bool = False) -> list[dict]:
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
    clip = []
    if feature_dir is not None:
        if (feature_dir / "meta.json").exists():
            clip.extend(_feature_results(connection, query, feature_dir, candidate_limit))
        if (feature_dir / "ids.json").exists() and (feature_dir / "vectors.npy").exists():
            # CLIP ViT-B/32 has weak Vietnamese support; search both the
            # original query and the lightweight English variant generated by
            # expand_query, then fuse both visual lists with lexical results.
            for visual_query in expand_query(query)[:2]:
                clip.extend(_official_clip_results(connection, visual_query, feature_dir, candidate_limit))
    lexical_lists = [channels[index] for index in range(0, len(channels), 2)]
    dense_lists = [channels[index] for index in range(1, len(channels), 2)]
    lexical = [row for ranked in lexical_lists for row in ranked]
    dense = [row for ranked in dense_lists for row in ranked]
    # Keep each retrieval channel independent. Flattening query variants into
    # one list lets duplicate hits from a single channel overpower the other
    # channels, especially for long Vietnamese queries.
    lists = (
        [[row["keyframe_id"] for row in ranked] for ranked in lexical_lists]
        + [[row["keyframe_id"] for row in ranked] for ranked in dense_lists]
        + [[row["keyframe_id"] for row in clip]]
    )
    lists = [items for items in lists if items]
    fused = rrf(lists)
    by_id = {row["keyframe_id"]: row for row in lexical + dense + clip}
    ordered = sorted(fused, key=fused.get, reverse=True)
    results = []
    seen_videos: set[str] = set()
    for keyframe_id in ordered:
        row = by_id[keyframe_id]
        row["retrieval_score"] = fused[keyframe_id]
        text = (row.get("text") or "").lower()
        query_tokens = set(re.findall(r"[\wÀ-ỹ]+", query.lower()))
        row["cooccurrence_score"] = sum(token in text for token in query_tokens) / max(1, len(query_tokens))
        row["temporal_score"] = temporal_score(row, query)
        row["retrieval_score"] += 0.08 * row["cooccurrence_score"] + 0.05 * row["temporal_score"]
        results.append(row)
        seen_videos.add(row["video_id"])
        if len(results) >= limit:
            break
    results.sort(key=lambda x: x["retrieval_score"], reverse=True)
    if preserve_video_coverage:
        # The official score uses R@1/R@5/R@20/R@50/R@100. Keep multiple
        # temporally separated frames from a relevant video so a correct
        # interval can still appear in a later cutoff.
        results = suppress_duplicates(results, per_video=max(10, limit // 2), min_gap=0.5)[:limit]
    else:
        results = suppress_duplicates(results, per_video=max(3, limit // 4))[:limit]
    for row in results:
        row["neighbors"] = neighbor_rows(connection, row["video_id"], row["keyframe_number"], neighbor_radius)
    return results


def export_submission(results: list[dict], output, limit: int = 100) -> None:
    import csv
    writer = csv.writer(output, lineterminator="\n")
    for row in results[:limit]: writer.writerow((row["video_id"], row["frame_id"]))
