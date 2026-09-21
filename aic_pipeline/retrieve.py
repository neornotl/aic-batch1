"""Lexical retrieval, RRF-compatible fusion, neighbors, and output export."""

from __future__ import annotations

import json
import os
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


_FTS_SIDECAR_PATH = Path(__file__).resolve().parents[1] / "work" / "aic_pipeline" / "keyframes_fts_folded.sqlite"
_FTS_MANIFEST_PATH = _FTS_SIDECAR_PATH.parent / "manifest.jsonl"
_folded_state: dict = {"checked": False, "ok": False, "con": None, "reason": ""}

_summary_state: dict = {"checked": False, "ok": False, "con": None, "reason": ""}
_transcript_state: dict = {"checked": False, "ok": False, "con": None, "reason": "", "path": ""}


def _search_transcript_segments(connection: sqlite3.Connection, transcript_db: Path | None,
                                text: str, limit: int = 80) -> list[dict]:
    """Use Deepgram utterances to select a video moment, then nearest BTC frame.

    Transcript evidence is deliberately a separate channel.  It can identify
    a spoken fact and its time, but it never invents a frame or changes the
    canonical ``frame_id`` used for submission.
    """
    if transcript_db is None:
        transcript_db = (Path(os.environ["TRANSCRIPT_SIDECAR_PATH"])
                         if os.environ.get("TRANSCRIPT_SIDECAR_PATH") else None)
    if transcript_db is None:
        return []
    state = _transcript_state
    if not state["checked"] or state.get("path") != str(Path(transcript_db).resolve()):
        old_connection = state.get("con")
        if old_connection is not None:
            try:
                old_connection.close()
            except Exception:
                pass
        state.update({"checked": True, "ok": False, "con": None,
                      "reason": "", "path": str(Path(transcript_db).resolve())})
        try:
            from .transcript_sidecar import verify
            report = verify(Path(transcript_db))
            if not report.get("ok"):
                raise RuntimeError("transcript sidecar verification failed")
            state["con"] = sqlite3.connect(
                f"file:{Path(transcript_db).resolve().as_posix()}?mode=ro", uri=True)
            state["ok"] = True
        except Exception as exc:
            state["reason"] = f"{type(exc).__name__}: {exc}"
    if not state["ok"]:
        return []
    expression = _fts_query(text)
    if not expression:
        return []
    try:
        rows = state["con"].execute(
            "SELECT s.segment_id,s.video_id,s.start_seconds,s.end_seconds,s.text,s.confidence,"
            "bm25(transcript_fts) AS bm25_score "
            "FROM transcript_fts f JOIN transcript_segments s ON s.segment_id=f.segment_id "
            "WHERE transcript_fts MATCH ? ORDER BY bm25_score LIMIT ?",
            (expression, max(1, int(limit))),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []
    columns = ["segment_id", "video_id", "start_seconds", "end_seconds",
               "transcript_match", "transcript_confidence", "transcript_bm25"]
    query_tokens = {
        token for token in re.findall(r"[\wÀ-ỹ]+", text.lower())
        if token not in _FTS_STOPWORDS and len(token) > 1
    }
    ranked_segments = []
    for raw in rows:
        segment_text = str(raw[4] or "").lower()
        coverage = (sum(token in segment_text for token in query_tokens)
                    / max(1, len(query_tokens)))
        # Long visual prompts often contain generic words that happen to be
        # spoken elsewhere. Require a little lexical agreement before a
        # transcript hit can influence frame ranking; short/name queries stay
        # permissive because one exact token can be decisive.
        if len(query_tokens) >= 3 and coverage < 0.25:
            continue
        ranked_segments.append((coverage, raw))
    ranked_segments.sort(key=lambda pair: (-pair[0], float(pair[1][6])))
    if not ranked_segments:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    frame_columns = [column[0] for column in connection.execute(
        "SELECT * FROM keyframes LIMIT 0").description]
    # Map each spoken segment to one actual official keyframe.  This is a
    # read-only lookup and falls back to the earliest frame when timestamps
    # are unavailable.
    for rank, (coverage, raw) in enumerate(ranked_segments, 1):
        segment = dict(zip(columns, raw))
        video_id = str(segment["video_id"])
        start = float(segment["start_seconds"])
        try:
            frame = connection.execute(
                "SELECT * FROM keyframes WHERE video_id=? "
                "ORDER BY CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END, "
                "ABS(COALESCE(timestamp, 0)-?) LIMIT 1", (video_id, start)
            ).fetchone()
        except sqlite3.Error:
            frame = None
        if frame is None:
            continue
        item = dict(zip(frame_columns, frame))
        key = str(item.get("keyframe_id"))
        # Keep the best segment when several utterances land on the same frame.
        if key in seen:
            continue
        seen.add(key)
        item.update(segment)
        item["transcript_rank"] = rank
        item["transcript_score"] = 1.0 / (rank ** 0.5)
        item["transcript_coverage"] = coverage
        item["retrieval_channel"] = "transcript"
        out.append(item)
    return out


def _search_video_summaries(connection, text: str, limit: int = 5) -> list[dict]:
    """Tra video-candidates trong summary sidecar. Fail-open: bat ky loi => []."""
    from .summary_sidecar import search as sidecar_search
    from .fts_sidecar.normalize import strip_diacritics as _sd
    from .summary_sidecar import verify as sidecar_verify

    st = _summary_state
    if os.environ.get("VIDEO_SUMMARY_SIDECAR", "0") != "1":
        return []
    path = os.environ.get("VIDEO_SUMMARY_SIDECAR_PATH", "")
    if not st.get("checked"):
        try:
            if not path or not Path(path).exists():
                raise FileNotFoundError(f"sidecar missing: {path}")
            rep = sidecar_verify(Path(path))
            if not rep.get("ok"):
                raise RuntimeError("sidecar verify failed")
            st["ok"] = True
            st["con"] = sqlite3.connect(
                f"file:{Path(path).as_posix()}?mode=ro", uri=True)
        except Exception as exc:
            st["ok"] = False
            st["reason"] = f"{type(exc).__name__}: {exc}"
        st["checked"] = True
    if not st["ok"]:
        return []
    expr = _fts_query(_sd(text))
    if not expr:
        return []
    try:
        rows = st["con"].execute(
            "SELECT video_id, bm25(videos_fts) AS sc FROM videos_fts "
            "WHERE videos_fts MATCH ? ORDER BY sc LIMIT ?",
            (expr, limit)).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for vid, sc in rows:
        out.append({"video_id": vid, "bm25_score": float(sc)})
    return out



def _search_fts_folded(connection: sqlite3.Connection, query: str, limit: int):
    """TEXT_FOLDED=1 route: normalized secondary FTS + batch keyframe fetch.

    Returns list[dict] in the same shape as search_fts, or None when the
    route is unavailable (flag off, artifact missing/stale/failed
    verification — including manifest SHA-256 mismatch — or any runtime
    error). Caller then falls back to the primary path. Never raises.
    """
    from .fts_sidecar.build import sha256_file as _sha256_file
    from .fts_sidecar.build import verify as sidecar_verify
    from .fts_sidecar.normalize import strip_diacritics
    from .fts_sidecar.router import folded_enabled

    st = _folded_state
    if not folded_enabled():
        return None
    if not st["checked"]:
        try:
            if not _FTS_SIDECAR_PATH.exists():
                raise FileNotFoundError(_FTS_SIDECAR_PATH)
            if not _FTS_MANIFEST_PATH.exists():
                raise FileNotFoundError(_FTS_MANIFEST_PATH)
            m_sha = _sha256_file(_FTS_MANIFEST_PATH)
            rep = sidecar_verify(_FTS_SIDECAR_PATH,
                                 expected_n_docs=None,
                                 manifest_sha256=m_sha)
            if not rep["ok"]:
                raise RuntimeError("sidecar verify failed (manifest/count/integrity): "
                                   + json.dumps({k: v for k, v in rep.items() if k != "meta"},
                                                ensure_ascii=False)[:300])
            n_primary = connection.execute(
                "SELECT COUNT(*) FROM keyframes").fetchone()[0]
            if n_primary != rep["n_docs_meta"]:
                raise RuntimeError(f"sidecar/primary count mismatch: "
                                   f"{rep['n_docs_meta']} vs {n_primary}")
            st["ok"] = True
            st["manifest_sha256"] = m_sha
            st["con"] = sqlite3.connect(
                rf"file:{_FTS_SIDECAR_PATH.as_posix()}?mode=ro", uri=True)
        except Exception as exc:
            st["ok"] = False
            st["reason"] = f"{type(exc).__name__}: {exc}"
        st["checked"] = True
    if not st["ok"]:
        return None
    expr = _fts_query(strip_diacritics(query))
    if not expr:
        return []
    rows = st["con"].execute(
        "SELECT kid, bm25(fts) AS sc FROM fts WHERE fts MATCH ? ORDER BY sc LIMIT ?",
        (expr, limit)).fetchall()
    kids = [kid for kid, _sc in rows]
    scores = {kid: float(sc) for kid, sc in rows}
    if not kids:
        return []
    placeholders = ",".join("?" for _ in kids)
    cursor = connection.execute(
        f"SELECT * FROM keyframes WHERE keyframe_id IN ({placeholders})", kids)
    columns = [c[0] for c in cursor.description]
    by_kid = {}
    for row in cursor.fetchall():
        item = dict(zip(columns, row))
        item["bm25_score"] = scores.get(item["keyframe_id"])
        item["sidecar_rank"] = None
        by_kid[item["keyframe_id"]] = item
    out = []
    for rank, kid in enumerate(kids):
        item = by_kid.get(kid)
        if item is None:
            continue
        item["sidecar_rank"] = rank
        out.append(item)
    return out


def search_fts(connection: sqlite3.Connection, query: str, limit: int = 300) -> list[dict]:
    folded_rows = None
    if os.environ.get("TEXT_FOLDED", "0") == "1":
        try:
            folded_rows = _search_fts_folded(connection, query, limit)
        except Exception:
            folded_rows = None
        if folded_rows is not None:
            return folded_rows
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


def _search_fts_in_videos_folded(
    connection: sqlite3.Connection,
    query: str,
    video_ids: list[str],
    limit: int,
    per_video: int,
) -> list[dict] | None:
    """Run the constrained pass against the verified folded sidecar.

    ``search_fts_in_videos`` is a rescue primitive, so it must not silently
    switch to a different corpus while ``TEXT_FOLDED`` is enabled.  ``None``
    means the sidecar is unavailable and lets the caller fail open to the
    primary FTS table; an empty list is a valid folded search with no hits.
    """
    try:
        # This also performs the existing manifest/count/integrity gate and
        # populates the shared read-only sidecar connection.
        _search_fts_folded(connection, query, 1)
    except Exception:
        return None
    state = _folded_state
    if not state.get("ok") or state.get("con") is None:
        return None
    from .fts_sidecar.normalize import strip_diacritics

    expression = _fts_query(strip_diacritics(query))
    if not expression or not video_ids:
        return []
    rows: list[dict] = []
    for start in range(0, len(video_ids), 300):
        video_batch = video_ids[start:start + 300]
        placeholders = ",".join("?" for _ in video_batch)
        try:
            cursor = state["con"].execute(
                f"""SELECT f.kid, bm25(fts) AS bm25_score
                    FROM fts AS f JOIN kid_map AS m ON m.kid=f.kid
                   WHERE fts MATCH ? AND m.vid IN ({placeholders})
                   ORDER BY bm25_score LIMIT ?""",
                [expression, *video_batch, max(limit * 3, 100)],
            )
        except sqlite3.OperationalError:
            return []
        rows.extend(
            {"keyframe_id": kid, "bm25_score": float(score)}
            for kid, score in cursor.fetchall()
        )
    if not rows:
        return []
    row_map = _rows_for_keyframe_ids(
        connection, [row["keyframe_id"] for row in rows]
    )
    selected: list[dict] = []
    per_video_count: dict[str, int] = defaultdict(int)
    for rank, match in enumerate(rows, 1):
        row = row_map.get(match["keyframe_id"])
        if row is None:
            continue
        video_id = row["video_id"]
        if per_video_count[video_id] >= per_video:
            continue
        item = dict(row)
        item["bm25_score"] = match["bm25_score"]
        item["sidecar_rank"] = rank - 1
        selected.append(item)
        per_video_count[video_id] += 1
        if len(selected) >= limit:
            break
    return selected


def search_fts_in_videos(connection: sqlite3.Connection, query: str, video_ids: list[str],
                         limit: int = 300, per_video: int = 8) -> list[dict]:
    """Find frame evidence inside videos selected by the timeline sidecar.

    A video-level match alone is not submittable. This constrained FTS pass
    injects concrete official frames from the selected videos back into the
    normal rank fusion while preventing one long video from taking every slot.
    """
    if os.environ.get("TEXT_FOLDED", "0") == "1":
        try:
            folded_rows = _search_fts_in_videos_folded(
                connection, query, video_ids, limit, per_video
            )
        except Exception:
            folded_rows = None
        if folded_rows is not None:
            return folded_rows
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


def _canonical_position(row: dict) -> tuple[float, float, float]:
    """Return the saved canonical order key for a keyframe.

    Official keyframe order is the primary coordinate. Timestamp and frame ID
    are deterministic tie-breakers only; no FPS-derived frame is ever made.
    """
    try:
        keyframe_number = float(row.get("keyframe_number"))
    except (TypeError, ValueError):
        keyframe_number = float("inf")
    try:
        timestamp = float(row.get("timestamp"))
    except (TypeError, ValueError):
        timestamp = float("inf")
    try:
        frame_id = float(row.get("frame_id"))
    except (TypeError, ValueError):
        frame_id = float("inf")
    return keyframe_number, timestamp, frame_id


def _frame_clue_coverage(query: str, row: dict) -> float:
    """Measure query-token coverage in frame-local fields only.

    ``text`` also contains repeated video metadata in this corpus. The rescue
    uses this small lexical signal to distinguish an actual frame clue from a
    title/description match without introducing a new model or index.
    """
    clue_query = clip_english_query(query) if any(ord(char) > 127 for char in query) else query
    raw_tokens = re.findall(r"[\wÀ-ỹ]+", clue_query.lower())
    tokens = [token for token in raw_tokens if token not in _FTS_STOPWORDS]
    if not tokens:
        tokens = raw_tokens
    frame_text = " ".join(
        str(row.get(field) or "") for field in (
            "caption", "ocr", "objects", "detector_classes", "object_entities", "asr"
        )
    ).lower()
    if not tokens or not frame_text:
        return 0.0
    return sum(token in frame_text for token in set(tokens)) / len(set(tokens))


def _video_consensus(
    connection: sqlite3.Connection,
    events: list[str],
    candidate_limit: int,
    context_limit: int = 300,
) -> tuple[list[dict], list[list[dict]]]:
    """Retrieve each event globally and aggregate evidence by video.

    The score is deliberately transparent: coverage is the primary sort key,
    followed by the sum of reciprocal best ranks and a small context boost.
    Context can nominate/boost a video but contributes no submit-ready frame.
    """
    event_rows: list[list[dict]] = []
    direct_ranks: dict[str, dict[int, int]] = defaultdict(dict)
    context_ranks: dict[str, list[int]] = defaultdict(list)
    for event_index, event in enumerate(events):
        rows = search_fts(connection, event, max(300, candidate_limit))
        event_rows.append(rows)
        for rank, row in enumerate(rows, 1):
            video_id = row.get("video_id")
            if video_id and event_index not in direct_ranks[video_id]:
                direct_ranks[video_id][event_index] = rank
        context_rows = search_video_context(
            connection, _fts_query(event), limit=max(1, context_limit)
        )
        for rank, row in enumerate(context_rows, 1):
            video_id = row.get("video_id")
            if video_id:
                context_ranks[video_id].append(rank)

    # A strict AND over event expressions is a cheap multi-event timeline
    # signal. It is intentionally used only for video selection; no context
    # row is returned as evidence.
    strict_context_ranks: dict[str, int] = {}
    strict_expression = " AND ".join(
        f"({_fts_query(event)})" for event in events if _fts_query(event)
    )
    if len(events) >= 2 and strict_expression:
        for rank, row in enumerate(
            search_video_context(connection, strict_expression, limit=max(1, context_limit)), 1
        ):
            video_id = row.get("video_id")
            if video_id and video_id not in strict_context_ranks:
                strict_context_ranks[video_id] = rank

    videos = set(direct_ranks) | set(context_ranks) | set(strict_context_ranks)
    candidates: list[dict] = []
    for video_id in videos:
        event_rank_map = direct_ranks.get(video_id, {})
        # Square-root reciprocal ranks reward broad support without letting a
        # single rank-1 event overwhelm two weaker events. A strict timeline
        # context hit is a secondary tie-break, not a frame score.
        reciprocal_rank = sum(1.0 / (rank ** 0.5) for rank in event_rank_map.values())
        strict_rank = strict_context_ranks.get(video_id)
        context_boost = (
            1.0 / (strict_rank ** 0.5) if strict_rank else 0.0
        ) + sum(
            0.05 / (rank ** 0.5) for rank in context_ranks.get(video_id, [])
        )
        candidates.append({
            "video_id": video_id,
            "coverage_count": len(event_rank_map),
            "event_count": len(events),
            "best_rank_by_event": {
                str(index + 1): rank for index, rank in sorted(event_rank_map.items())
            },
            "reciprocal_rank_sum": reciprocal_rank,
            "strict_context_rank": strict_rank,
            "video_context_boost": context_boost,
            "score": reciprocal_rank + context_boost,
        })
    candidates.sort(key=lambda item: (
        -item["coverage_count"], -item["score"], -item["reciprocal_rank_sum"],
        item["video_id"],
    ))
    for rank, item in enumerate(candidates, 1):
        item["video_rank"] = rank
    return candidates, event_rows


def _best_temporal_sequence(
    event_candidates: list[list[dict]], minimum_event_gap: int = 0
) -> dict | None:
    """Choose the highest-scoring strictly ordered event sequence by DP."""
    if not event_candidates or any(not candidates for candidates in event_candidates):
        return None
    # Each state keeps deterministic tie-break values. When several repeated
    # frames have the same frame-local clue coverage, choose the earliest
    # canonical sequence; BM25 rank remains the small secondary signal.
    states: list[list[tuple[float, float, float, list[dict]]]] = []
    for event_index, candidates in enumerate(event_candidates):
        current: list[tuple[float, float, float, list[dict]]] = []
        for candidate in candidates:
            local_rank = max(1, int(candidate.get("rescue_rank", 1)))
            score = float(candidate.get("rescue_score", 1.0 / local_rank))
            position = _canonical_position(candidate)[0]
            rank_signal = 1.0 / (local_rank ** 0.5)
            best_state = (score, position, rank_signal, [candidate])
            if event_index:
                previous = [
                    state for state in states[-1]
                    if _canonical_position(state[3][-1]) < _canonical_position(candidate)
                    and (
                        minimum_event_gap <= 0
                        or _canonical_position(candidate)[0]
                        - _canonical_position(state[3][-1])[0] >= minimum_event_gap
                    )
                ]
                if previous:
                    prior = max(
                        previous,
                        key=lambda state: (state[0], -state[1], state[2]),
                    )
                    best_state = (
                        prior[0] + score,
                        prior[1] + position,
                        prior[2] + rank_signal,
                        prior[3] + [candidate],
                    )
            current.append(best_state)
        states.append(current)
    if not states[-1]:
        return None
    score, _position_sum, _rank_signal, rows = max(
        states[-1], key=lambda state: (state[0], -state[1], state[2])
    )
    if len(rows) != len(event_candidates):
        return None
    return {"score": score, "rows": rows}


def search_trake_sequence(
    connection: sqlite3.Connection,
    events: list[str],
    *,
    candidate_limit: int = 300,
    candidate_video_limit: int = 5,
    event_candidate_limit: int = 50,
    neighbor_radius: int = 2,
    minimum_event_gap: int = 0,
) -> dict:
    """Run the lightweight TRAKE retrieval rescue.

    Global event retrieval is used only to discover and rank candidate videos.
    Every event is then searched independently inside each of the top videos;
    a small DP selects an ordered sequence, and canonical neighbors are added
    as evidence metadata after representative frames are selected.
    """
    cleaned_events = [" ".join(str(event).split()) for event in events if str(event).strip()]
    if not cleaned_events:
        return {"events": [], "video_candidates": [], "sequences": []}
    candidate_limit = max(300, int(candidate_limit))
    candidate_video_limit = max(1, int(candidate_video_limit))
    event_candidate_limit = max(5, int(event_candidate_limit))
    consensus, _global_event_rows = _video_consensus(
        connection, cleaned_events, candidate_limit
    )
    selected_candidates: list[dict] = []
    sequence_candidates: list[dict] = []
    for candidate in consensus[:candidate_video_limit]:
        video_id = candidate["video_id"]
        per_event: list[list[dict]] = []
        for event_index, event in enumerate(cleaned_events, 1):
            rows = search_fts_in_videos(
                connection, event, [video_id],
                limit=event_candidate_limit, per_video=event_candidate_limit,
            )
            ranked_rows: list[dict] = []
            for local_rank, row in enumerate(rows, 1):
                item = dict(row)
                item["event_index"] = event_index
                item["fts_rank"] = local_rank
                clue_coverage = _frame_clue_coverage(event, item)
                item["frame_clue_coverage"] = clue_coverage
                ranked_rows.append(item)
            # The raw FTS order contains repeated video-title metadata. Put
            # actual frame-local clues first, then use canonical time and raw
            # FTS rank as deterministic tie-breakers.
            ranked_rows.sort(key=lambda item: (
                -float(item["frame_clue_coverage"]),
                _canonical_position(item),
                int(item["fts_rank"]),
            ))
            for rescue_rank, item in enumerate(ranked_rows, 1):
                item["rescue_rank"] = rescue_rank
                # Exact frame-local clues tie on quality and are resolved by
                # the canonical sequence. Partial clues retain a small rank
                # signal so lexical relevance remains useful.
                coverage = float(item["frame_clue_coverage"])
                item["rescue_score"] = coverage + (
                    0.01 / (int(item["fts_rank"]) ** 0.5) if coverage < 1.0 else 0.0
                )
            per_event.append(ranked_rows)
        entry = dict(candidate)
        entry["event_candidates"] = per_event
        sequence = _best_temporal_sequence(per_event, minimum_event_gap)
        entry["sequence"] = None
        if sequence is not None:
            representatives = sequence["rows"]
            neighbor_frame_ids: list[list[int]] = []
            for row in representatives:
                neighbors = neighbor_rows(
                    connection, video_id, int(row["keyframe_number"]), neighbor_radius
                )
                neighbor_frame_ids.append([
                    int(neighbor["frame_id"]) for neighbor in neighbors
                ])
            entry["sequence"] = {
                "score": float(sequence["score"]),
                "selection_score": float(sequence["score"]) + float(candidate["score"]),
                "frame_ids": [int(row["frame_id"]) for row in representatives],
                "keyframe_numbers": [int(row["keyframe_number"]) for row in representatives],
                "neighbor_frame_ids": neighbor_frame_ids,
                "temporal_order_valid": all(
                    _canonical_position(representatives[index])
                    < _canonical_position(representatives[index + 1])
                    for index in range(len(representatives) - 1)
                ),
            }
            sequence_candidates.append(entry)
        selected_candidates.append(entry)
    sequence_candidates.sort(key=lambda item: (
        -float(item["sequence"]["selection_score"]), item["video_rank"]
    ))
    return {
        "events": cleaned_events,
        "video_candidates": selected_candidates,
        "sequences": sequence_candidates,
        "selected_sequence": sequence_candidates[0]["sequence"] if sequence_candidates else None,
        "selected_video_id": (
            sequence_candidates[0]["video_id"] if sequence_candidates else None
        ),
    }


def search_qa_evidence_video_first(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 100,
    candidate_limit: int = 300,
    candidate_video_limit: int = 5,
    per_video_limit: int = 30,
) -> dict:
    """Retrieve QA evidence by selecting videos before local frame ranking."""
    candidate_limit = max(300, int(candidate_limit))
    global_rows = search_fts(connection, query, candidate_limit)
    direct: dict[str, dict] = {}
    for rank, row in enumerate(global_rows, 1):
        video_id = row.get("video_id")
        if video_id and video_id not in direct:
            direct[video_id] = {"best_rank": rank, "reciprocal_rank": 1.0 / rank}
    context: dict[str, int] = {}
    for rank, row in enumerate(
        search_video_context(connection, _fts_query(query), limit=candidate_limit), 1
    ):
        video_id = row.get("video_id")
        if video_id and video_id not in context:
            context[video_id] = rank
    candidates = []
    for video_id in set(direct) | set(context):
        direct_info = direct.get(video_id, {})
        context_rank = context.get(video_id)
        context_boost = 0.05 / (context_rank ** 0.5) if context_rank else 0.0
        candidates.append({
            "video_id": video_id,
            "best_global_rank": direct_info.get("best_rank"),
            "reciprocal_rank": direct_info.get("reciprocal_rank", 0.0),
            "video_context_rank": context_rank,
            "video_context_boost": context_boost,
            "score": direct_info.get("reciprocal_rank", 0.0) + context_boost,
        })
    candidates.sort(key=lambda item: (-item["score"], item["video_id"]))
    for rank, candidate in enumerate(candidates, 1):
        candidate["video_rank"] = rank
    rescued_rows: list[dict] = []
    selected_candidates = candidates[:max(1, int(candidate_video_limit))]
    for candidate in selected_candidates:
        local_rows = search_fts_in_videos(
            connection, query, [candidate["video_id"]],
            limit=max(1, int(per_video_limit)), per_video=max(1, int(per_video_limit)),
        )
        for local_rank, row in enumerate(local_rows, 1):
            item = dict(row)
            item["video_candidate_rank"] = candidate["video_rank"]
            item["local_rank"] = local_rank
            item["rescue_score"] = 1.0 / local_rank + 0.05 * candidate["score"]
            rescued_rows.append(item)
    rescued_rows.sort(key=lambda row: (
        -float(row["rescue_score"]), row["video_candidate_rank"], row["local_rank"]
    ))
    return {
        "query": query,
        "video_candidates": selected_candidates,
        "rows": rescued_rows[:max(1, int(limit))],
        "global_rows": global_rows,
    }


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


def search(connection: sqlite3.Connection, query: str, limit: int = 20, candidate_limit: int = 300, neighbor_radius: int = 2, dense_dir: Path | None = None, feature_dir: Path | None = None, expansions: list[str] | None = None, preserve_video_coverage: bool = False, include_neighbors: bool = True, transcript_db: Path | None = None) -> list[dict]:
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

    # Deepgram is a video-moment discovery channel, not a replacement for
    # visual evidence.  Match utterances, map them to official BTC frames by
    # timestamp, and fuse those frames at ordinary RRF weight.  Keeping this
    # channel opt-in preserves the frozen SAFE release when no sidecar is set.
    transcript_lists: list[list[dict]] = []
    transcript_variants = list(dict.fromkeys([variants[0], *( [en_q] if en_q else [])]))
    for variant in transcript_variants:
        rows = _search_transcript_segments(connection, transcript_db, variant,
                                            limit=min(candidate_limit, 120))
        if rows:
            transcript_lists.append(rows)

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
    transcript = [row for ranked in transcript_lists for row in ranked]
    # Keep each retrieval channel independent. Flattening query variants into
    # one list lets duplicate hits from a single channel overpower the other
    # channels, especially for long Vietnamese queries.
    # Weight CLIP higher for KIS: visual evidence should dominate lexical
    # noise, especially when the query is Vietnamese but captions are English.
    lists = (
        [[row["keyframe_id"] for row in ranked] for ranked in lexical_lists]
        + [[row["keyframe_id"] for row in ranked] for ranked in dense_lists]
        + [[row["keyframe_id"] for row in ranked] for ranked in transcript_lists]
        + [[row["keyframe_id"] for row in ranked] for ranked in clip_lists] * 3
    )
    lists = [items for items in lists if items]
    fused = rrf(lists)
    clip = [row for ranked in clip_lists for row in ranked]
    by_id = {row["keyframe_id"]: row for row in lexical + dense + transcript + clip}
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
        if "transcript_rank" in row:
            # Coverage is more informative than raw FTS rank: Deepgram
            # utterances are short and BM25 can rank a generic one-token hit
            # ahead of a semantically complete sentence.
            row["retrieval_score"] += (
                0.06 * float(row.get("transcript_coverage", 0.0))
                + 0.02 * float(row.get("transcript_score", 0.0))
            )
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
