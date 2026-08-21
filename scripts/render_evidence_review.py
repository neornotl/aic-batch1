"""Render compact contact sheets from an evidence-review plan.

Each source JPEG is fetched with :class:`RangeZip`, pasted into its contact
sheet, and immediately deleted.  This keeps runner storage bounded by one
image plus the finished sheets, rather than by source videos or ZIP archives.
"""

from __future__ import annotations

import argparse
import html
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from aic_pipeline.range_zip import RangeZip


ARCHIVE_ROOT = "https://aic-data.ledo.io.vn/Keyframes_{}.zip"
CELL_WIDTH = 320
CELL_HEIGHT = 220
HEADER_HEIGHT = 34
COLUMNS = 4


def _archive_code(video_id: str) -> str:
    return video_id.split("_", 1)[0]


def _selected(plan: dict, names: set[str]) -> list[dict]:
    queries = plan.get("queries", [])
    if not names:
        return queries
    unknown = names - {str(query.get("id")) for query in queries}
    if unknown:
        raise ValueError(f"Unknown query IDs: {', '.join(sorted(unknown))}")
    return [query for query in queries if query.get("id") in names]


def _cell_label(candidate: dict, frame: dict) -> str:
    return f"#{candidate['rank']}  {candidate['video_id']}  f={frame['frame_id']}"


def _sheet_for_query(query: dict, output: Path, scratch: Path, archives: dict[str, RangeZip]) -> dict:
    cells = [(candidate, frame) for candidate in query.get("candidates", []) for frame in candidate.get("frames", [])]
    rows = max(1, (len(cells) + COLUMNS - 1) // COLUMNS)
    canvas = Image.new("RGB", (COLUMNS * CELL_WIDTH, rows * (CELL_HEIGHT + HEADER_HEIGHT)), "white")
    draw = ImageDraw.Draw(canvas)
    failures: list[str] = []
    for position, (candidate, frame) in enumerate(cells):
        x = position % COLUMNS * CELL_WIDTH
        y = position // COLUMNS * (CELL_HEIGHT + HEADER_HEIGHT)
        label = _cell_label(candidate, frame)
        temporary = scratch / f"{query['id']}-{position}.jpg"
        try:
            code = _archive_code(candidate["video_id"])
            archive = archives.setdefault(code, RangeZip(ARCHIVE_ROOT.format(code)))
            archive.download(frame["member"], temporary)
            with Image.open(temporary) as source:
                thumb = ImageOps.fit(source.convert("RGB"), (CELL_WIDTH, CELL_HEIGHT), method=Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x, y))
        except Exception as error:  # Keep other candidates reviewable.
            failures.append(f"{label}: {error}")
            draw.rectangle((x, y, x + CELL_WIDTH, y + CELL_HEIGHT), fill="#fee2e2")
            draw.text((x + 8, y + 8), "DOWNLOAD FAILED", fill="#991b1b")
        finally:
            # Never retain an original keyframe after its thumbnail was made.
            temporary.unlink(missing_ok=True)
        draw.rectangle((x, y + CELL_HEIGHT, x + CELL_WIDTH, y + CELL_HEIGHT + HEADER_HEIGHT), fill="#111827")
        draw.text((x + 6, y + CELL_HEIGHT + 9), label, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)
    return {"id": query["id"], "kind": query["kind"], "text": query["text"], "sheet": output.name,
            "frames": len(cells), "failures": failures}


def _write_index(items: list[dict], output: Path) -> None:
    sections = []
    for item in items:
        failures = "".join(f"<li>{html.escape(value)}</li>" for value in item["failures"])
        sections.append(
            f"<section><h2>{html.escape(item['id'])} <small>({html.escape(item['kind'])})</small></h2>"
            f"<p>{html.escape(item['text'])}</p><img src=\"sheets/{html.escape(item['sheet'])}\" "
            f"alt=\"{html.escape(item['id'])} contact sheet\"><ul>{failures}</ul></section>"
        )
    output.write_text(
        "<!doctype html><meta charset=utf-8><title>AIC evidence review</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1400px}img{max-width:100%;border:1px solid #ccc}"
        "small{color:#666}</style><h1>AIC evidence review</h1>" + "\n".join(sections),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", default="", help="Comma-separated query IDs; blank means all")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    names = {value.strip() for value in args.queries.split(",") if value.strip()}
    selected = _selected(plan, names)
    args.output.mkdir(parents=True, exist_ok=True)
    sheets = args.output / "sheets"
    archives: dict[str, RangeZip] = {}
    with tempfile.TemporaryDirectory(prefix="aic-review-") as tmp:
        scratch = Path(tmp)
        items = [_sheet_for_query(query, sheets / f"{query['id']}.jpg", scratch, archives) for query in selected]
    (args.output / "review.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_index(items, args.output / "index.html")
    print(json.dumps({"queries": len(items), "frames": sum(item["frames"] for item in items)}))


if __name__ == "__main__":
    main()
