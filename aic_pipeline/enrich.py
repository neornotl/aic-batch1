"""Resumable Terra enrichment for captions, OCR, and object interpretation."""
from __future__ import annotations
import json
from pathlib import Path
from .terra import TerraAdapter

def enrich_manifest(manifest: Path, output: Path, limit: int | None = None) -> int:
    adapter=TerraAdapter(); output.parent.mkdir(parents=True, exist_ok=True); done={}
    if output.exists():
        for line in output.open(encoding="utf-8"):
            try: done[json.loads(line)["keyframe_id"]]=True
            except (json.JSONDecodeError, KeyError): pass
    count=0
    with manifest.open(encoding="utf-8") as source, output.open("a", encoding="utf-8") as target:
        for line in source:
            if limit is not None and count >= limit: break
            item=json.loads(line)
            if item["keyframe_id"] in done: continue
            prompt=json.dumps({"caption":item.get("caption"),"ocr":item.get("ocr"),"objects":item.get("objects"),"detector":item.get("object_entities")}, ensure_ascii=False)
            result=adapter.complete("Normalize this keyframe metadata. Return {caption,ocr,objects:[...],actions:[...],location} and preserve uncertain text.\n"+prompt)
            if result:
                item["caption"]=str(result.get("caption") or item.get("caption", "")); item["ocr"]=str(result.get("ocr") or item.get("ocr", ""))
                item["objects"]=" ".join(map(str, result.get("objects") or item.get("objects", "").split()))
                item["text"]=" ".join(str(item.get(k,"")) for k in ("video_id","caption","ocr","objects","object_entities","asr"))
            target.write(json.dumps(item, ensure_ascii=False)+"\n"); target.flush(); count += 1
    return count
