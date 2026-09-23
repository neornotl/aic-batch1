"""Offline-first AIC2026 competition RC server.

The RC keeps Batch 1 canonical keyframes read-only and searches the isolated
Batch 2 evidence-safe video/event database. Batch 2 results are deliberately
not submission-capable until the user supplies and validates the keyframe DB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


APP_ROOT = Path(__file__).resolve().parents[1]
_repo_candidates = [p for p in APP_ROOT.parents if (p / "work" / "aic_pipeline" / "keyframes.sqlite").exists()]
_default_canonical = (
    APP_ROOT / "data" / "keyframes.sqlite"
    if (APP_ROOT / "data" / "keyframes.sqlite").exists()
    else (_repo_candidates[0] / "work" / "aic_pipeline" / "keyframes.sqlite" if _repo_candidates else APP_ROOT / "data" / "keyframes.sqlite")
)
BATCH2_DB = Path(
    os.environ.get("AIC_BATCH2_DB", APP_ROOT / "data" / "BATCH2_VIDEO_DATABASE.sqlite")
).expanduser()
CANONICAL_DB = Path(
    os.environ.get("AIC_CANONICAL_DB", _default_canonical)
).expanduser()
STATIC_ROOT = APP_ROOT / "app"

_thread_local = threading.local()
_STOPWORDS = {
    "a", "an", "and", "are", "at", "by", "for", "from", "in", "is", "it", "of",
    "on", "the", "to", "with", "after", "before", "then", "that", "this",
    "các", "cảnh", "có", "của", "đang", "được", "khi", "là", "một", "những",
    "người", "này", "phần", "sau", "sau đó", "sẽ", "trên", "trong", "và", "với",
}
_TEMPORAL = (
    "before", "after", "then", "later", "while", "first", "next", "finally",
    "trước", "sau", "sau đó", "rồi", "trong khi", "đầu tiên", "cuối cùng",
)
_QUESTION = (
    "who", "what", "where", "when", "why", "how", "which",
    "ai", "cái gì", "gì", "ở đâu", "khi nào", "tại sao", "vì sao", "như thế nào",
)


def _connect(path: Path) -> sqlite3.Connection:
    uri = "file:" + path.resolve().as_posix() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def batch2_connection() -> sqlite3.Connection:
    connection = getattr(_thread_local, "batch2", None)
    if connection is None:
        connection = _connect(BATCH2_DB)
        _thread_local.batch2 = connection
    return connection


def canonical_connection() -> sqlite3.Connection | None:
    if not CANONICAL_DB.is_file():
        return None
    connection = getattr(_thread_local, "canonical", None)
    if connection is None:
        connection = _connect(CANONICAL_DB)
        _thread_local.canonical = connection
    return connection


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[\wÀ-ỹ]+", query.casefold())
    selected: list[str] = []
    for token in tokens:
        if token not in _STOPWORDS and token not in selected:
            selected.append(token)
    if not selected:
        selected = tokens
    return " OR ".join('"' + token.replace('"', "") + '"' for token in selected[:32])


def route_task(query: str, requested: str = "AUTO") -> str:
    requested = requested.upper()
    if requested in {"KIS", "QA", "TRAKE"}:
        return requested
    lowered = query.casefold()
    if any(token in lowered for token in _TEMPORAL) and len(re.findall(r"\s+", query)) >= 2:
        return "TRAKE"
    if any(token in lowered for token in _QUESTION):
        return "QA"
    return "KIS"


def _row_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def _batch2_video_results(connection: sqlite3.Connection, match: str, limit: int) -> list[dict]:
    try:
        rows = connection.execute(
            """
            SELECT video_id, bm25(video_fts) AS rank_score
            FROM video_fts
            WHERE video_fts MATCH ?
            ORDER BY rank_score
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
    except sqlite3.Error:
        return []
    results: list[dict] = []
    for rank, row in enumerate(rows, 1):
        detail = connection.execute(
            """
            SELECT video_id, batch, overview, topics_json, people_json, objects_json,
                   actions_json, locations_json, visible_text_json, spoken_content_json,
                   story_sequence_json, aliases_json, asset_status, keyframe_status
            FROM videos WHERE video_id = ?
            """,
            (row["video_id"],),
        ).fetchone()
        if detail is None:
            continue
        item = _row_dict(detail)
        item.update(
            {
                "result_type": "VIDEO",
                "source": "BATCH_02",
                "rank": rank,
                "score": round(float(-row["rank_score"]), 6),
                "submission_capable": False,
                "canonical_mapping_status": "PENDING_BATCH2_KEYFRAME_DATABASE",
            }
        )
        results.append(item)
    return results


def _batch2_event_results(connection: sqlite3.Connection, match: str, limit: int) -> list[dict]:
    try:
        rows = connection.execute(
            """
            SELECT event_id, video_id, bm25(event_fts) AS rank_score
            FROM event_fts
            WHERE event_fts MATCH ?
            ORDER BY rank_score
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
    except sqlite3.Error:
        return []
    results: list[dict] = []
    for rank, row in enumerate(rows, 1):
        detail = connection.execute(
            """
            SELECT event_id, video_id, start_sec, end_sec, anchor_sec, description,
                   actions_json, objects_json, people_json, roles_json, scene, location,
                   grounded_aliases_json, temporal_relations_json, source_frame_id,
                   source_keyframe_id, canonical_mapping_status
            FROM events WHERE event_id = ?
            """,
            (row["event_id"],),
        ).fetchone()
        if detail is None:
            continue
        item = _row_dict(detail)
        item.update(
            {
                "result_type": "EVENT",
                "source": "BATCH_02",
                "rank": rank,
                "score": round(float(-row["rank_score"]), 6),
                "submission_capable": False,
                "asset_status": "BACKFILLING",
                "canonical_mapping_status": "PENDING_BATCH2_KEYFRAME_DATABASE",
            }
        )
        results.append(item)
    return results


def _canonical_results(connection: sqlite3.Connection | None, match: str, limit: int) -> list[dict]:
    if connection is None:
        return []
    try:
        rows = connection.execute(
            """
            SELECT k.keyframe_id, k.video_id, k.keyframe_number, k.frame_id,
                   k.timestamp, k.path, k.caption, k.ocr,
                   bm25(keyframes_fts) AS rank_score
            FROM keyframes_fts
            JOIN keyframes AS k ON k.rowid = keyframes_fts.rowid
            WHERE keyframes_fts MATCH ?
            ORDER BY rank_score
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {
            **_row_dict(row),
            "result_type": "KEYFRAME",
            "source": "BATCH_01_CANONICAL",
            "rank": rank,
            "score": round(float(-row["rank_score"]), 6),
            "submission_capable": True,
            "canonical_mapping_status": "CANONICAL_SQLITE",
            "asset_status": "AVAILABLE" if row["path"] else "UNKNOWN",
        }
        for rank, row in enumerate(rows, 1)
    ]


def search(query: str, requested_task: str = "AUTO", limit: int = 20) -> dict:
    query = query.strip()
    if not query:
        return {"ok": False, "error": "Query cannot be empty."}
    match = _fts_query(query)
    if not match:
        return {"ok": False, "error": "Query contains no searchable terms."}
    task = route_task(query, requested_task)
    limit = max(1, min(int(limit), 100))
    started = time.perf_counter()
    batch2 = batch2_connection()
    canonical = canonical_connection()
    if task == "TRAKE":
        batch2_results = _batch2_event_results(batch2, match, limit)
    else:
        batch2_results = _batch2_video_results(batch2, match, max(limit // 2, 5))
        batch2_results += _batch2_event_results(batch2, match, max(limit // 2, 5))
    canonical_results = _canonical_results(canonical, match, limit)
    all_results = canonical_results + batch2_results
    all_results.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            item.get("source", ""),
            item.get("result_type", ""),
            item.get("video_id", ""),
        )
    )
    results = all_results[:limit]
    for rank, item in enumerate(results, 1):
        item["rank"] = rank
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    notes = [
        "Batch 2 evidence-safe video/event retrieval is active.",
        "Batch 2 keyframe mapping is pending the user-supplied keyframe database.",
        "Inference-only content is excluded from the primary path.",
    ]
    if not canonical_results:
        notes.append("No Batch 1 canonical keyframe result matched this query.")
    return {
        "ok": True,
        "query": query,
        "task": task,
        "requested_task": requested_task.upper(),
        "match_expression": match,
        "latency_ms": elapsed_ms,
        "result_count": len(results),
        "results": results,
        "notes": notes,
    }


def video_detail(video_id: str) -> dict | None:
    connection = batch2_connection()
    row = connection.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
    if row is None:
        return None
    events = connection.execute(
        """
        SELECT event_id, start_sec, end_sec, anchor_sec, description, actions_json,
               objects_json, people_json, roles_json, scene, location,
               grounded_aliases_json, temporal_relations_json, source_frame_id,
               source_keyframe_id, canonical_mapping_status
        FROM events WHERE video_id=? ORDER BY start_sec, event_id
        """,
        (video_id,),
    ).fetchall()
    return {
        "source": "BATCH_02",
        "video": _row_dict(row),
        "events": [_row_dict(event) for event in events],
        "keyframes": [],
        "submission_capable": False,
        "canonical_mapping_status": "PENDING_BATCH2_KEYFRAME_DATABASE",
    }


def canonical_video_detail(video_id: str) -> dict | None:
    connection = canonical_connection()
    if connection is None:
        return None
    row = connection.execute(
        """
        SELECT video_id, count(*) AS keyframe_count
        FROM keyframes WHERE video_id = ? GROUP BY video_id
        """,
        (video_id,),
    ).fetchone()
    if row is None:
        return None
    context = connection.execute(
        "SELECT text, sample_count FROM video_context WHERE video_id=? LIMIT 1",
        (video_id,),
    ).fetchone()
    keyframes = connection.execute(
        """
        SELECT keyframe_id, video_id, keyframe_number, frame_id, timestamp,
               path, caption, ocr
        FROM keyframes WHERE video_id=?
        ORDER BY keyframe_number, keyframe_id
        LIMIT 24
        """,
        (video_id,),
    ).fetchall()
    return {
        "source": "BATCH_01_CANONICAL",
        "video": {
            "video_id": video_id,
            "batch": "BATCH_01",
            "overview": context["text"] if context else "",
            "sample_count": context["sample_count"] if context else None,
            "keyframe_count": row["keyframe_count"],
            "asset_status": "AVAILABLE",
            "keyframe_status": "CANONICAL_SQLITE",
        },
        "events": [],
        "keyframes": [_row_dict(item) for item in keyframes],
        "submission_capable": True,
        "canonical_mapping_status": "CANONICAL_SQLITE",
    }


def health() -> dict:
    batch2_ready = BATCH2_DB.is_file()
    canonical_ready = CANONICAL_DB.is_file()
    counts = {}
    if batch2_ready:
        con = batch2_connection()
        counts["batch2_videos"] = con.execute("SELECT count(*) FROM videos").fetchone()[0]
        counts["batch2_events"] = con.execute("SELECT count(*) FROM events").fetchone()[0]
        counts["batch2_keyframe_status"] = dict(
            con.execute(
                "SELECT keyframe_db_status,count(*) FROM asset_status GROUP BY keyframe_db_status"
            ).fetchall()
        )
    if canonical_ready:
        con = canonical_connection()
        counts["batch1_keyframes"] = con.execute("SELECT count(*) FROM keyframes").fetchone()[0]
        counts["batch1_videos"] = con.execute("SELECT count(distinct video_id) FROM keyframes").fetchone()[0]
    return {
        "ok": batch2_ready,
        "state": "READY" if batch2_ready else "BLOCKED",
        "canonical_state": "AVAILABLE_READ_ONLY" if canonical_ready else "UNAVAILABLE",
        "batch2_database": str(BATCH2_DB),
        "canonical_database": str(CANONICAL_DB),
        "keyframe_ingest": "PENDING_USER_SUPPLIED_DATABASE",
        "inference_primary_path": "EXCLUDED",
        "counts": counts,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AIC2026AppRC/1.0"

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._send_json({"ok": False, "error": "static asset unavailable"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(health())
            return
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            task = params.get("task", ["AUTO"])[0]
            limit = params.get("limit", ["20"])[0]
            try:
                payload = search(query, task, int(limit))
                self._send_json(payload, 200 if payload.get("ok") else 400)
            except (sqlite3.Error, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return
        if parsed.path.startswith("/api/video/"):
            video_id = unquote(parsed.path.removeprefix("/api/video/"))
            detail = video_detail(video_id)
            if detail is None:
                detail = canonical_video_detail(video_id)
            if detail is None:
                self._send_json({"ok": False, "error": "video not found"}, 404)
            else:
                self._send_json({"ok": True, **detail})
            return
        if parsed.path in {"/", "/index.html"}:
            self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self._send_json(health())
            return
        self._send_json({"ok": False, "error": "not found"}, 404)

    def log_message(self, format: str, *args: object) -> None:
        print("[AIC2026]", format % args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("AIC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIC_PORT", "8765")))
    args = parser.parse_args()
    status = health()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AIC2026_APP_RC_READY http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
