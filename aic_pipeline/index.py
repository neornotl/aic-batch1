"""SQLite FTS index for keyframe retrieval, with optional dense vectors later."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS keyframes (
  rowid INTEGER PRIMARY KEY,
  keyframe_id TEXT UNIQUE NOT NULL,
  video_id TEXT NOT NULL,
  keyframe_number INTEGER NOT NULL,
  frame_id INTEGER NOT NULL,
  timestamp REAL,
  fps REAL,
  path TEXT NOT NULL,
  caption TEXT,
  ocr TEXT,
  objects TEXT,
  detector_classes TEXT,
  object_entities TEXT,
  asr TEXT,
  media_info TEXT,
  text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS keyframes_fts USING fts5(
  keyframe_id UNINDEXED, text, content='keyframes', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);
"""


def build_index(manifest: Path, database: Path) -> int:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(SCHEMA)
        connection.execute("DROP TABLE IF EXISTS keyframes_fts")
        connection.execute("DROP TABLE IF EXISTS keyframes")
        connection.executescript(SCHEMA)
        connection.execute("DELETE FROM keyframes_fts")
        connection.execute("DELETE FROM keyframes")
        rows = []
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                rows.append((
                    item["keyframe_id"], item["video_id"], item["keyframe_number"],
                    item["frame_id"], item.get("timestamp"), item.get("fps"),
                    item["path"], item.get("caption", ""), item.get("ocr", ""),
                     item.get("objects", ""), item.get("detector_classes", ""), item.get("object_entities", ""),
                     item.get("asr", ""), item.get("media_info", ""),
                    item.get("text", ""),
                ))
                if len(rows) >= 2000:
                    connection.executemany(
                        "INSERT INTO keyframes(keyframe_id,video_id,keyframe_number,frame_id,timestamp,fps,path,caption,ocr,objects,detector_classes,object_entities,asr,media_info,text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        rows,
                    )
                    rows.clear()
        if rows:
            connection.executemany(
                "INSERT INTO keyframes(keyframe_id,video_id,keyframe_number,frame_id,timestamp,fps,path,caption,ocr,objects,detector_classes,object_entities,asr,media_info,text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        connection.execute("INSERT INTO keyframes_fts(keyframes_fts) VALUES ('rebuild')")
        connection.commit()
        return int(connection.execute("SELECT COUNT(*) FROM keyframes").fetchone()[0])
    finally:
        connection.close()
