from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")

from cir_pipeline.retrieval.faiss_index import FaissIndex, save_mapping
from cir_pipeline.retrieval.search import search_index


def test_faiss_index_roundtrip_and_reference_filter(tmp_path):
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype="float32")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = FaissIndex(dim=2, index_type="flat_ip")
    index.add(embeddings)
    index_path = tmp_path / "index.faiss"
    mapping_path = tmp_path / "mapping.json"
    index.save(index_path)
    save_mapping(
        [
            {"faiss_id": 0, "image_name": "ref", "image_path": "ref.png", "split": "val"},
            {"faiss_id": 1, "image_name": "target", "image_path": "target.png", "split": "val"},
            {"faiss_id": 2, "image_name": "other", "image_path": "other.png", "split": "val"},
        ],
        mapping_path,
    )
    query = np.array([[1.0, 0.0]], dtype="float32")
    results = search_index(index_path, mapping_path, query, top_k=2, reference_image_name="ref")
    assert results[0]["image_name"] == "other"
    assert all(row["image_name"] != "ref" for row in results)
