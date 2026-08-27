"""Small end-to-end Vertex/GCS smoke test used by GitHub Actions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    video_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    bucket_name = os.environ.get("GCS_BUCKET")
    api_key = os.environ.get("VERTEX_API_KEY")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "aic2026-drive-ingest")
    if not bucket_name or not api_key:
        raise SystemExit("GCS_BUCKET and VERTEX_API_KEY are required")

    from google import genai
    from google.cloud import storage
    from google.genai import types

    storage_client = storage.Client(project=project)
    blob_name = "aic-batch1-summary/_smoke/smoke.mp4"
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    client = genai.Client(
        vertexai=True,
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1"),
    )
    try:
        blob.upload_from_filename(str(video_path), content_type="video/mp4", timeout=120)
        response = client.models.generate_content(
            model=os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"),
            contents=[
                types.Part.from_uri(
                    file_uri=f"gs://{bucket_name}/{blob_name}",
                    mime_type="video/mp4",
                ),
                "Describe this short test video in one sentence.",
            ],
        )
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Vertex returned an empty response")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "status": "ok",
            "video_id": "SMOKE",
            "source_member": "synthetic/smoke.mp4",
            "model": os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"),
            "summary": text,
            "confidence": "high",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Vertex/GCS smoke passed", flush=True)
    finally:
        try:
            blob.delete(timeout=120)
        except Exception as cleanup_error:  # noqa: BLE001
            print(f"GCS smoke cleanup warning: {cleanup_error!r}", flush=True)


if __name__ == "__main__":
    main()
