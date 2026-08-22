"""Command line entry points for the AIC keyframe MVP."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .index import build_index, ensure_indexes, refresh_search_text
from .context import build_video_context_index
from .manifest import build_manifest
from .retrieve import export_submission, search
from .dense import build_dense_index
from .dense import build_tfidf_index
from .rerank import local_rerank, remote_judge, should_judge
from .evaluate import evaluate
from .enrich import enrich_manifest
from .terra import TerraAdapter
from .submit import package_submission, run_query_file, write_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "aic-batch1-results-final"
DEFAULT_MAPS = ROOT / "pilot_data" / "map-keyframes" / "map-keyframes"
DEFAULT_OBJECTS = ROOT / "work" / "data" / "objects" / "objects"
DEFAULT_MEDIA = ROOT / "work" / "data" / "media-info" / "media-info"
DEFAULT_WORK = ROOT / "work" / "aic_pipeline"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    manifest.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    manifest.add_argument("--objects", type=Path, default=DEFAULT_OBJECTS)
    manifest.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    manifest.add_argument("--asr", type=Path)
    manifest.add_argument("--output", type=Path, default=DEFAULT_WORK / "manifest.jsonl")
    index = sub.add_parser("index")
    index.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "manifest.jsonl")
    index.add_argument("--database", type=Path, default=DEFAULT_WORK / "keyframes.sqlite")
    context = sub.add_parser("context")
    context.add_argument("--database", type=Path, default=DEFAULT_WORK / "keyframes.sqlite")
    context.add_argument("--max-samples", type=int, default=160)
    refresh = sub.add_parser("refresh-text")
    refresh.add_argument("--database", type=Path, default=DEFAULT_WORK / "keyframes.sqlite")
    dense = sub.add_parser("dense")
    dense.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "manifest.jsonl")
    dense.add_argument("--output", type=Path, default=DEFAULT_WORK / "dense")
    dense.add_argument("--device", default=None)
    dense.add_argument("--backend", choices=("bge", "tfidf"), default="tfidf")
    enrich = sub.add_parser("enrich")
    enrich.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "manifest.jsonl")
    enrich.add_argument("--output", type=Path, default=DEFAULT_WORK / "manifest.enriched.jsonl")
    enrich.add_argument("--limit", type=int)
    query = sub.add_parser("search")
    query.add_argument("text")
    query.add_argument("--database", type=Path, default=DEFAULT_WORK / "keyframes.sqlite")
    query.add_argument("--limit", type=int, default=20)
    query.add_argument("--dense", type=Path,
                       help="Optional dense index; frame/context FTS is the fast default")
    query.add_argument("--clip", type=Path)
    query.add_argument("--json", action="store_true")
    query.add_argument("--rerank", action="store_true")
    query.add_argument("--judge", choices=("off", "auto", "always"), default="off")
    query.add_argument("--terra-expand", action="store_true")
    query.add_argument("--submission", type=Path)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--ground-truth", type=Path, required=True)
    submission = sub.add_parser("submission")
    submission.add_argument("--results", type=Path, required=True, help="JSONL ranked results")
    submission.add_argument("--output", type=Path, required=True)
    batch = sub.add_parser("batch")
    batch.add_argument("--queries", type=Path, required=True)
    batch.add_argument("--database", type=Path, default=DEFAULT_WORK / "keyframes.sqlite")
    batch.add_argument("--dense", type=Path,
                       help="Optional dense index; omit for fast frame/context FTS")
    batch.add_argument("--clip", type=Path)
    batch.add_argument("--candidate-limit", type=int, default=120)
    batch.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                       help="Independent FTS workers; use 1 when a remote judge is enabled")
    batch.add_argument("--output", type=Path, required=True)
    pack = sub.add_parser("package")
    pack.add_argument("--csv-dir", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        print(build_manifest(args.results, args.maps, args.objects, args.output, args.media, args.asr))
    elif args.command == "index":
        print(build_index(args.manifest, args.database))
    elif args.command == "context":
        connection = sqlite3.connect(args.database)
        try:
            ensure_indexes(connection)
            print(build_video_context_index(connection, args.max_samples))
        finally:
            connection.close()
    elif args.command == "refresh-text":
        connection = sqlite3.connect(args.database)
        try:
            print(refresh_search_text(connection))
        finally:
            connection.close()
    elif args.command == "dense":
        if args.backend == "bge":
            print(build_dense_index(args.manifest, args.output, device=args.device))
        else:
            print(build_tfidf_index(args.manifest, args.output))
    elif args.command == "enrich":
        print(enrich_manifest(args.manifest, args.output, args.limit))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.predictions, args.ground_truth), indent=2))
    elif args.command == "submission":
        rows = []
        with args.results.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): rows.append(json.loads(line))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as output:
            export_submission(rows, output)
        print(len(rows))
    elif args.command == "batch":
        args.output.mkdir(parents=True, exist_ok=True)
        # One upfront write so parallel workers only ever read.
        with sqlite3.connect(args.database) as setup:
            ensure_indexes(setup)
        terra_adapter = TerraAdapter()
        if not terra_adapter.key:
            terra_adapter = None
        query_files = sorted(args.queries.glob("*.txt"))

        def run_one(query_file: Path) -> tuple[Path, list[list[str]]]:
            connection = sqlite3.connect(args.database)
            try:
                rows = run_query_file(
                    query_file, connection, args.dense, terra_adapter=terra_adapter,
                    feature_dir=args.clip, candidate_limit=args.candidate_limit,
                )
                return query_file, rows
            finally:
                connection.close()

        workers = max(1, args.workers)
        # A remote judge can rate-limit or share a non-thread-safe client; FTS
        # retrieval itself is read-only and benefits from parallel workers.
        if terra_adapter is not None:
            workers = 1
        if workers == 1:
            completed = (run_one(query_file) for query_file in query_files)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                completed = list(executor.map(run_one, query_files))
        for query_file, rows in completed:
            output = args.output / f"{query_file.stem}.csv"
            write_csv(rows, output)
            print(f"{query_file.name}: {len(rows)} rows")
    elif args.command == "package":
        print(json.dumps(package_submission(args.csv_dir, args.output), indent=2))
    else:
        connection = sqlite3.connect(args.database)
        try:
            ensure_indexes(connection)
            pool_limit = max(args.limit, 300 if args.rerank else args.limit)
            expansions = None
            if args.terra_expand:
                terra = TerraAdapter()
                answer = terra.complete("Expand this video retrieval query into up to 5 Vietnamese-English search variants. Return {\"queries\": [strings]}. Query: " + args.text)
                expansions = answer.get("queries", []) if isinstance(answer, dict) else []
            results = search(connection, args.text, pool_limit, dense_dir=args.dense, feature_dir=args.clip, expansions=expansions)
            if args.rerank:
                results = local_rerank(args.text, results)[:args.limit]
            else:
                results = results[:args.limit]
            if args.judge != "off" and should_judge(results, always=args.judge == "always"):
                results = remote_judge(args.text, results)
            if args.submission:
                args.submission.parent.mkdir(parents=True, exist_ok=True)
                with args.submission.open("w", encoding="utf-8", newline="") as output:
                    export_submission(results, output)
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                for rank, row in enumerate(results, 1):
                    print(f"#{rank} {row['video_id']} frame={row['frame_id']} t={row['timestamp']} {row['path']}")
                    print(f"   {row['caption'][:220]}")
        finally:
            connection.close()


if __name__ == "__main__":
    main()
