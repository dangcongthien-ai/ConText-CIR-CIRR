"""FAISS index wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cir_pipeline.utils.io import read_json, write_json


def _require_faiss():
    try:
        import faiss

        return faiss
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "FAISS is required for retrieval. Install faiss-cpu, or faiss-gpu matching CUDA."
        ) from exc


@dataclass
class FaissSearchResult:
    scores: np.ndarray
    indices: np.ndarray


class FaissIndex:
    """Build, save, load, and search an inner-product FAISS index."""

    def __init__(self, dim: int, index_type: str = "flat_ip", hnsw_m: int = 32) -> None:
        faiss = _require_faiss()
        self.dim = dim
        self.index_type = index_type
        if index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported FAISS index_type: {index_type}")

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings shape [N, {self.dim}], got {embeddings.shape}")
        self.index.add(embeddings)

    def search(self, queries: np.ndarray, top_k: int) -> FaissSearchResult:
        queries = np.asarray(queries, dtype="float32")
        scores, indices = self.index.search(queries, top_k)
        return FaissSearchResult(scores=scores, indices=indices)

    def save(self, path: str | Path) -> None:
        faiss = _require_faiss()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        faiss = _require_faiss()
        index = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj.index = index
        obj.dim = int(index.d)
        obj.index_type = "loaded"
        return obj


def save_mapping(mapping: list[dict[str, Any]], path: str | Path) -> None:
    write_json(mapping, path)


def load_mapping(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected FAISS mapping to be a list: {path}")
    return data
