import sqlite3
import json
import tempfile
import unittest
import struct
from pathlib import Path

from aic_pipeline.retrieve import _fts_query, _official_clip_keyframe_id, neighbor_rows, rrf
from aic_pipeline.evaluate import evaluate
from aic_pipeline.range_zip import RangeZip, RemoteEntry


class RetrievalTests(unittest.TestCase):
    def test_fts_query_quotes_tokens(self):
        self.assertEqual(_fts_query("xe đỏ"), '"xe" OR "đỏ"')

    def test_rrf_rewards_agreement(self):
        scores = rrf([["a", "b"], ["a", "c"]])
        self.assertGreater(scores["a"], scores["b"])

    def test_official_clip_archive_uses_one_based_db_key(self):
        self.assertEqual(_official_clip_keyframe_id("L21_V024", 0), "L21_V024:1")
        self.assertEqual(_official_clip_keyframe_id("L21_V024", 203), "L21_V024:204")

    def test_neighbor_rows_are_bounded(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            """
            CREATE TABLE keyframes (
              keyframe_id TEXT, video_id TEXT, keyframe_number INTEGER
            );
            INSERT INTO keyframes VALUES ('v:1','v',1),('v:2','v',2),
              ('v:3','v',3),('other:3','other',3);
            """
        )
        rows = neighbor_rows(connection, "v", 2, 1)
        self.assertEqual([row["keyframe_number"] for row in rows], [1, 2, 3])

    def test_evaluation_accepts_frame_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.jsonl"
            predictions = root / "predictions.jsonl"
            truth.write_text(json.dumps({"query_id": "q1", "accepted": [{"video_id": "v", "min_frame": 10, "max_frame": 20}]}) + "\n", encoding="utf-8")
            predictions.write_text(json.dumps({"query_id": "q1", "rank": 2, "video_id": "v", "frame_id": 15}) + "\n", encoding="utf-8")
            metrics = evaluate(predictions, truth)
            self.assertEqual(metrics["recall@1"], 0.0)
            self.assertEqual(metrics["recall@5"], 1.0)
            self.assertEqual(metrics["mrr"], 0.5)

    def test_range_zip_uses_local_header_extra_length(self):
        archive = object.__new__(RangeZip)
        # Central and local ZIP headers may legitimately use different extra
        # field lengths. The local one determines where file bytes begin.
        local = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, 0, 0, 0, 0, 10, 10, 7, 13)
        archive._request = lambda start, end: local  # type: ignore[method-assign]
        entry = RemoteEntry("keyframes/V/1.jpg", 10, 10, 100, 1, 1, 0)
        self.assertEqual(archive._data_start(entry), 150)


if __name__ == "__main__":
    unittest.main()
