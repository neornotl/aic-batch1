"""Optional BGE-M3 dense index stored as a memory-mapped NumPy matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_dense_index(manifest: Path, output_dir: Path, model_name: str = "BAAI/bge-m3", batch_size: int = 16, device: str | None = None) -> int:
    from sentence_transformers import SentenceTransformer

    output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    ids: list[str] = []
    texts: list[str] = []
    checkpoint = output_dir / "checkpoint.json"
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            ids.append(item["keyframe_id"])
            texts.append(item.get("text", ""))

    matrix_path = output_dir / "vectors.npy"
    checkpoint = output_dir / "checkpoint.json"
    start = 0
    if checkpoint.exists() and matrix_path.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("model") == model_name and state.get("count") == len(ids):
            start = int(state.get("done", 0))
    vectors = None
    for offset in range(start, len(texts), batch_size):
        batch = model.encode(
            texts[offset:offset + batch_size],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        if vectors is None:
            if matrix_path.exists() and start:
                vectors = np.lib.format.open_memmap(matrix_path, mode="r+")
            else:
                vectors = np.lib.format.open_memmap(
                    matrix_path, mode="w+", dtype="float32", shape=(len(texts), batch.shape[1])
                )
        vectors[offset:offset + len(batch)] = batch
        vectors.flush()
        checkpoint.write_text(json.dumps({"model": model_name, "count": len(ids), "done": offset + len(batch)}), encoding="utf-8")
    dim = int(vectors.shape[1]) if vectors is not None else int(np.load(matrix_path, mmap_mode="r").shape[1])
    if vectors is not None:
        del vectors
    checkpoint.unlink(missing_ok=True)
    (output_dir / "ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (output_dir / "meta.json").write_text(json.dumps({"kind": "bge", "model": model_name, "count": len(ids), "dim": dim}), encoding="utf-8")
    return len(ids)


def build_tfidf_index(manifest: Path, output_dir: Path, max_features: int = 120_000) -> int:
    """Fast CPU-safe semantic-ish fallback using word and character TF-IDF."""
    from scipy.sparse import save_npz
    from sklearn.feature_extraction.text import TfidfVectorizer

    output_dir.mkdir(parents=True, exist_ok=True)
    ids, texts = [], []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            ids.append(item["keyframe_id"])
            texts.append(item.get("text", ""))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=max_features,
        sublinear_tf=True, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    save_npz(output_dir / "vectors_tfidf.npz", matrix)
    import pickle
    with (output_dir / "vectorizer.pkl").open("wb") as handle:
        pickle.dump(vectorizer, handle)
    (output_dir / "ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (output_dir / "meta.json").write_text(json.dumps({"kind": "tfidf", "count": len(ids), "dim": int(matrix.shape[1])}), encoding="utf-8")
    return len(ids)


class DenseIndex:
    def __init__(self, directory: Path):
        self.directory = directory
        self.vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        self.ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))

    def search(self, vector: np.ndarray, limit: int = 300) -> list[tuple[str, float]]:
        scores = np.asarray(self.vectors @ vector, dtype="float32")
        limit = min(limit, len(scores))
        indices = np.argpartition(scores, -limit)[-limit:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [(self.ids[int(index)], float(scores[int(index)])) for index in indices]


class TfidfIndex:
    def __init__(self, directory: Path):
        import pickle
        from scipy.sparse import load_npz
        self.matrix = load_npz(directory / "vectors_tfidf.npz")
        self.ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        with (directory / "vectorizer.pkl").open("rb") as handle:
            self.vectorizer = pickle.load(handle)

    def search(self, query: str, limit: int = 300) -> list[tuple[str, float]]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        limit = min(limit, len(scores))
        indices = np.argpartition(scores, -limit)[-limit:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [(self.ids[int(index)], float(scores[int(index)])) for index in indices if scores[int(index)] > 0]
