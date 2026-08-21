"""Build and query the official per-video CLIP feature archive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_index(feature_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = []
    ids = []
    for path in sorted(feature_dir.glob("*.npy")):
        matrix = np.asarray(np.load(path), dtype="float32")
        vectors.append(matrix)
        ids.extend(f"{path.stem}:{number}" for number in range(len(matrix)))
    joined = np.concatenate(vectors, axis=0)
    joined /= np.linalg.norm(joined, axis=1, keepdims=True).clip(min=1e-8)
    np.save(output_dir / "vectors.npy", joined.astype("float16"))
    (output_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    return {"count": len(ids), "dim": int(joined.shape[1])}
