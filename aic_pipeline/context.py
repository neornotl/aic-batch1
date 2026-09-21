"""Video-level context index built from already-captioned BTC keyframes.

Frame captions are good at locating a visual detail, but an AIC prompt often
describes a sequence spanning several cuts. This module creates a compact
timeline document per video so retrieval can first identify videos that cover
several parts of the description, then use the regular keyframe index to find
the evidence frame.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable


CONTEXT_SCHEMA = """
CREATE TABLE IF NOT EXISTS video_context (
  rowid INTEGER PRIMARY KEY,
  video_id TEXT UNIQUE NOT NULL,
  sample_count INTEGER NOT NULL,
  text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS video_context_fts USING fts5(
  video_id UNINDEXED,
  text,
  content='video_context',
  content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);
"""


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _dedupe_fragments(values: Iterable[str], maximum: int) -> list[str]:
    """Keep distinct timeline observations without bloating the FTS document."""
    kept: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        if not text:
            continue
        # Captions from adjacent official frames can differ only in whitespace
        # or a timestamp. Dedupe that noise but retain the first occurrence.
        key = re.sub(r"\d+", "#", text.lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append(text)
        if len(kept) >= maximum:
            break
    return kept


def _timeline_text(rows: list[sqlite3.Row], max_samples: int) -> tuple[int, str]:
    """Create a bounded, chronological document for one video."""
    if not rows:
        return 0, ""
    # Uniform sampling preserves early, middle and late events. A separate
    # de-duplication pass below makes repeated broadcast shots cheap.
    stride = max(1, (len(rows) + max_samples - 1) // max_samples)
    sampled = rows[::stride]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    fragments = []
    for row in sampled:
        observation = " ".join(
            value for value in (
                _clean(row["caption"]),
                _clean(row["ocr"]),
                _clean(row["objects"]),
                _clean(row["detector_classes"]),
                _clean(row["object_entities"]),
            )
            if value
        )
        if observation:
            fragments.append(f"at keyframe {row['keyframe_number']}: {observation}")
    compact = _dedupe_fragments(fragments, max_samples)
    return len(compact), " . ".join(compact)


def build_video_context_index(connection: sqlite3.Connection, max_samples: int = 160) -> int:
    """(Re)build the small sidecar FTS table from the keyframe table.

    ``max_samples`` is deliberately bounded: it keeps the index laptop-safe
    while retaining temporal coverage for long videos.
    """
    if max_samples < 8:
        raise ValueError("max_samples must be at least 8")
    connection.executescript(
        "DROP TABLE IF EXISTS video_context_fts; DROP TABLE IF EXISTS video_context;"
    )
    connection.executescript(CONTEXT_SCHEMA)
    connection.row_factory = sqlite3.Row
    cursor = connection.execute(
        """SELECT video_id, keyframe_number, caption, ocr, objects,
                  detector_classes, object_entities
           FROM keyframes ORDER BY video_id, keyframe_number"""
    )
    current_video = None
    rows: list[sqlite3.Row] = []
    count = 0
    for row in cursor:
        if current_video is not None and row["video_id"] != current_video:
            sample_count, text = _timeline_text(rows, max_samples)
            connection.execute(
                "INSERT INTO video_context(video_id, sample_count, text) VALUES (?,?,?)",
                (current_video, sample_count, text),
            )
            count += 1
            rows = []
        current_video = row["video_id"]
        rows.append(row)
    if current_video is not None:
        sample_count, text = _timeline_text(rows, max_samples)
        connection.execute(
            "INSERT INTO video_context(video_id, sample_count, text) VALUES (?,?,?)",
            (current_video, sample_count, text),
        )
        count += 1
    connection.execute("INSERT INTO video_context_fts(video_context_fts) VALUES ('rebuild')")
    connection.commit()
    return count


def search_video_context(connection: sqlite3.Connection, expression: str, limit: int = 60) -> list[dict]:
    """Return the most relevant videos, or no rows when the sidecar is absent."""
    if not expression:
        return []
    try:
        cursor = connection.execute(
            """SELECT v.video_id, v.sample_count, bm25(video_context_fts) AS context_bm25
               FROM video_context_fts f JOIN video_context v ON v.rowid=f.rowid
               WHERE video_context_fts MATCH ?
               ORDER BY context_bm25 LIMIT ?""",
            (expression, limit),
        )
    except sqlite3.OperationalError:
        return []
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
