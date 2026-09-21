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
from .operator import decorate_results, format_operator_table
from .validator import validate_input


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
    query.add_argument("--transcript", type=Path,
                       help="Optional verified Deepgram transcript sidecar (SQLite)")
    query.add_argument("--candidate-limit", type=int, default=300,
                       help="Pre-deduplication candidate pool; release default is 300")
    query.add_argument("--preserve-video-coverage", action="store_true",
                       help="Keep temporally separated candidates from relevant videos")
    query.add_argument("--json", action="store_true")
    query.add_argument("--rerank", action="store_true")
    query.add_argument("--judge", choices=("off", "auto", "always"), default="off")
    query.add_argument("--terra-expand", action="store_true")
    query.add_argument("--submission", type=Path)
    query.add_argument("--neighbors", type=int, default=2,
                       help="Canonical keyframe neighbors on each side")
    query.add_argument("--query-type", default="auto",
                       choices=("auto", "KIS_SCENE", "KIS_TEXT_NUMBER", "QA", "TRAKE_MULTI_EVENT"))
    query.add_argument("--keyframe-root", type=Path,
                       help="Optional root used to show a resolved local keyframe path")
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
    batch.add_argument("--transcript", type=Path,
                       help="Optional verified Deepgram transcript sidecar (SQLite)")
    batch.add_argument("--candidate-limit", type=int, default=300,
                       help="Pre-deduplication candidate pool; release default is 300")
    batch.add_argument("--preserve-video-coverage", action="store_true",
                       help="Keep temporally separated candidates from relevant videos")
    batch.add_argument("--retrieval-rescue", action="store_true",
                       help="Experimental TRAKE/QA video-first rescue; release default is off")
    batch.add_argument("--trake-rescue-v2", action="store_true",
                       help="Experimental TRAKE-only joint video/sequence rescue; default is off")
    batch.add_argument("--trake-topk-v3", action="store_true",
                       help="Experimental TRAKE-only ranked full-answer sequence rescue; default is off")
    batch.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                       help="Independent FTS workers; use 1 when a remote judge is enabled")
    batch.add_argument("--output", type=Path, required=True)
    vs_imp = sub.add_parser("import-video-summaries")
    vs_imp.add_argument("--jsonl", type=Path, nargs="+", required=True)
    vs_imp.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "manifest.jsonl")
    vs_imp.add_argument("--database", type=Path, default=DEFAULT_WORK / "keyframes.sqlite")
    vs_imp.add_argument("--output", type=Path, default=DEFAULT_WORK / "video_summaries.sqlite")
    vs_ver = sub.add_parser("verify-video-summaries")
    vs_ver.add_argument("--sidecar", type=Path, default=DEFAULT_WORK / "video_summaries.sqlite")
    vs_ver.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "manifest.jsonl")
    vs_ver.add_argument("--database", type=Path, default=DEFAULT_WORK / "keyframes.sqlite")
    transcript = sub.add_parser("transcript-index",
                                help="Build a verified Deepgram transcript sidecar")
    transcript.add_argument("--jsonl", type=Path, nargs="+", required=True,
                            help="Deepgram JSONL files or glob-expanded paths")
    transcript.add_argument("--output", type=Path,
                            default=DEFAULT_WORK / "deepgram_transcripts.sqlite")
    transcript_verify = sub.add_parser("verify-transcript-index")
    transcript_verify.add_argument("--sidecar", type=Path,
                                   default=DEFAULT_WORK / "deepgram_transcripts.sqlite")
    pack = sub.add_parser("package")
    pack.add_argument("--csv-dir", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--database", type=Path,
                      help="Canonical keyframes.sqlite; required for strict release mode")
    pack.add_argument("--config", type=Path,
                      help="Strict release config with required_files and format rules")
    release = sub.add_parser("release", help="Fail-closed, canonical-identity-checked ZIP release")
    release.add_argument("--csv-dir", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--database", type=Path, required=True)
    release.add_argument("--config", type=Path, required=True)
    validate = sub.add_parser("validate", help="Validate a candidate CSV, directory, or ZIP")
    validate.add_argument("input", type=Path)
    validate.add_argument("--database", type=Path,
                          help="Optional keyframes.sqlite for official identity checks")
    validate.add_argument("--config", type=Path,
                          help="JSON config for verified task-specific arity/quota/order rules")
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
                    preserve_video_coverage=args.preserve_video_coverage,
                    transcript_db=args.transcript,
                    retrieval_rescue=args.retrieval_rescue,
                    trake_rescue_v2=args.trake_rescue_v2,
                    trake_topk_v3=args.trake_topk_v3,
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
    elif args.command == "import-video-summaries":
        from .summary_sidecar import build_atomic
        import sqlite3 as _s3
        _con = _s3.connect(f"file:{args.database.as_posix()}?mode=ro", uri=True)
        try:
            known = {r[0] for r in _con.execute("SELECT DISTINCT video_id FROM keyframes")}
        finally:
            _con.close()
        try:
            rep = build_atomic(args.jsonl, args.output, known)
            print(json.dumps(rep, ensure_ascii=False, indent=2))
            sys.exit(0 if rep.get("status") == "ok" else 1)
        except ValueError as exc:
            print(json.dumps({"status": "failed", "error": str(exc)},
                             ensure_ascii=False, indent=2))
            sys.exit(1)
    elif args.command == "verify-video-summaries":
        from .summary_sidecar import verify as verify_summaries
        rep = verify_summaries(args.sidecar,
                               known_video_ids=None)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        sys.exit(0 if rep.get("ok") else 1)
    elif args.command == "transcript-index":
        from .transcript_sidecar import build_atomic
        rep = build_atomic(args.jsonl, args.output)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        sys.exit(0 if rep.get("status") == "ok" else 1)
    elif args.command == "verify-transcript-index":
        from .transcript_sidecar import verify
        rep = verify(args.sidecar)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        sys.exit(0 if rep.get("ok") else 1)
    elif args.command == "package":
        if args.database is None or args.config is None:
            print(json.dumps({
                "ok": False,
                "status": "FAILED",
                "errors": ["package requires --database and --config; use the strict release contract"],
                "published": False,
            }, indent=2))
            sys.exit(2)
        from .release import ReleaseValidationError, build_release
        try:
            print(json.dumps(build_release(args.csv_dir, args.output, args.database, args.config), indent=2))
        except ReleaseValidationError as exc:
            print(json.dumps(exc.report, ensure_ascii=False, indent=2))
            sys.exit(1)
    elif args.command == "release":
        from .release import ReleaseValidationError, build_release
        try:
            print(json.dumps(build_release(args.csv_dir, args.output, args.database, args.config), indent=2))
        except ReleaseValidationError as exc:
            print(json.dumps(exc.report, ensure_ascii=False, indent=2))
            sys.exit(1)
    elif args.command == "validate":
        report = validate_input(args.input, args.database, args.config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        sys.exit(0 if report["ok"] else 1)
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
            candidate_limit = max(300, args.candidate_limit, args.limit)
            results = search(
                connection, args.text, pool_limit,
                candidate_limit=candidate_limit,
                dense_dir=args.dense, feature_dir=args.clip, expansions=expansions,
                preserve_video_coverage=args.preserve_video_coverage,
                transcript_db=args.transcript,
                include_neighbors=False,
            )
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
            operator_rows = decorate_results(
                connection, args.text, results, neighbor_radius=args.neighbors,
                query_hint=args.query_type, keyframe_root=args.keyframe_root,
            )
            if args.json:
                print(json.dumps(operator_rows, ensure_ascii=False, indent=2))
            else:
                print(format_operator_table(operator_rows))
        finally:
            connection.close()


if __name__ == "__main__":
    main()
