"""Build a submission-safe keyframe manifest from AIC artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterator


def _json_response(value: str) -> dict:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _frame_number(path: str) -> int:
    match = re.search(r"/(\d+)\.jpg$", path.replace("\\", "/"))
    if not match:
        raise ValueError(f"Cannot parse keyframe number from {path}")
    return int(match.group(1))


def load_maps(map_dir: Path) -> dict[tuple[str, int], dict]:
    maps: dict[tuple[str, int], dict] = {}
    for csv_path in sorted(map_dir.glob("*.csv")):
        video_id = csv_path.stem
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                n = int(row["n"])
                maps[(video_id, n)] = {
                    "frame_id": int(row["frame_idx"]),
                    "timestamp": float(row["pts_time"]),
                    "fps": float(row["fps"]),
                }
    return maps


def _object_text(objects_root: Path, video_id: str, number: int) -> tuple[str, str]:
    path = objects_root / video_id / f"{number:03d}.json"
    if not path.exists():
        return "", ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", ""
    names = data.get("detection_class_names", [])
    entities = data.get("detection_class_entities", [])
    return (" ".join(str(name).replace("/m/", "class_") for name in names[:100]),
            " ".join(str(name) for name in entities[:100]))


def iter_records(results_dir: Path, map_dir: Path, objects_root: Path, media_root: Path | None = None,
                 asr_root: Path | None = None) -> Iterator[dict]:
    maps = load_maps(map_dir)
    seen: set[str] = set()
    for result_path in sorted(results_dir.glob("results_*.jsonl")):
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                path = str(item.get("path", "")).replace("\\", "/")
                if not path or path in seen:
                    continue
                match = re.search(r"keyframes/([^/]+)/([0-9]+)\.jpg$", path)
                if not match:
                    continue
                video_id, number_text = match.groups()
                number = int(number_text)
                mapping = maps.get((video_id, number), {})
                if not mapping:
                    # Keep the record searchable while making missing mapping data explicit.
                    mapping = {"frame_id": number, "timestamp": None, "fps": None}
                response = _json_response(item.get("response", ""))
                caption = str(response.get("scene_description", ""))
                ocr = str(response.get("on_screen_text", ""))
                objects = response.get("objects", [])
                if not isinstance(objects, list):
                    objects = [objects]
                object_text = " ".join(str(value) for value in objects)
                detector_text, object_entities = _object_text(objects_root, video_id, number)
                media = {}
                if media_root:
                    media_path = media_root / f"{video_id}.json"
                    if media_path.exists():
                        try:
                            media = json.loads(media_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            media = {}
                asr = ""
                if asr_root:
                    for candidate in (asr_root / f"{video_id}.json", asr_root / f"{video_id}.jsonl"):
                        if candidate.exists():
                            try:
                                raw = candidate.read_text(encoding="utf-8")
                                asr = " ".join(str(x.get("text", x)) if isinstance(x, dict) else str(x)
                                              for x in (json.loads(raw) if candidate.suffix == ".json" else [json.loads(x) for x in raw.splitlines()]))
                            except (OSError, json.JSONDecodeError):
                                pass
                            break
                record = {
                    # Keyframe number identifies the archived image; source frame_id
                    # can legitimately collide when the map contains duplicate PTS.
                    "keyframe_id": f"{video_id}:{number}",
                    "video_id": video_id,
                    "keyframe_number": number,
                    "frame_id": mapping.get("frame_id", number),
                    "timestamp": mapping.get("timestamp"),
                    "fps": mapping.get("fps"),
                    "path": path,
                    "caption": caption,
                    "ocr": ocr,
                    "objects": object_text,
                    "detector_classes": detector_text,
                    "object_entities": object_entities,
                    "asr": asr,
                    "media_info": json.dumps(media, ensure_ascii=False, sort_keys=True),
                }
                record["text"] = " ".join(
                    value for value in (
                        video_id,
                        caption,
                        ocr,
                        object_text,
                        detector_text, object_entities, asr, json.dumps(media, ensure_ascii=False),
                    ) if value
                )
                seen.add(path)
                yield record


def build_manifest(results_dir: Path, map_dir: Path, objects_root: Path, output: Path,
                   media_root: Path | None = None, asr_root: Path | None = None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in iter_records(results_dir, map_dir, objects_root, media_root, asr_root):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
