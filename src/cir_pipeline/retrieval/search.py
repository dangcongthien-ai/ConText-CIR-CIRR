"""Query encoding and FAISS search helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cir_pipeline.retrieval.faiss_index import FaissIndex, load_mapping
from cir_pipeline.utils.io import write_json


def encode_query_image_text(
    model,
    query_image: str | Path,
    text: str,
    device,
    transform=None,
    *,
    backbone: str | None = None,
    transform_name: str = "clip",
    target_pad_ratio: float = 1.25,
) -> np.ndarray:
    import torch
    from PIL import Image

    from cir_pipeline.data.transforms import build_clip_transform

    resolved_transform = transform or build_clip_transform(
        None,
        False,
        backbone,
        transform_name,
        target_pad_ratio,
    )
    image = Image.open(query_image).convert("RGB")
    tensor = resolved_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_query(tensor, [text]).detach().cpu().numpy().astype("float32")
    norm = np.linalg.norm(embedding, axis=1, keepdims=True).clip(min=1e-12)
    return embedding / norm


def search_index(
    index_path: str | Path,
    mapping_path: str | Path,
    query_embedding: np.ndarray,
    top_k: int,
    reference_image_name: str | None = None,
) -> list[dict[str, Any]]:
    index = FaissIndex.load(index_path)
    mapping = load_mapping(mapping_path)
    return search_loaded_index(index, mapping, query_embedding, top_k, reference_image_name)


def search_loaded_index(
    index: FaissIndex,
    mapping: list[dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int,
    reference_image_name: str | None = None,
) -> list[dict[str, Any]]:
    search_k = top_k + 10 if reference_image_name else top_k
    result = index.search(query_embedding, search_k)
    rows: list[dict[str, Any]] = []
    for score, faiss_id in zip(result.scores[0], result.indices[0]):
        if faiss_id < 0:
            continue
        item = mapping[int(faiss_id)]
        if reference_image_name and item["image_name"] == reference_image_name:
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "image_name": item["image_name"],
                "image_path": item["image_path"],
                "score": float(score),
            }
        )
        if len(rows) >= top_k:
            break
    return rows


def save_retrieval_results(results: list[dict[str, Any]], path: str | Path) -> None:
    write_json({"results": results}, path)
