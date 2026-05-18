"""Encode CIRR gallery images and build FAISS files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from cir_pipeline.config import AppConfig
from cir_pipeline.data.cirr_dataset import CIRRDataset
from cir_pipeline.engine.evaluator import encode_gallery
from cir_pipeline.retrieval.faiss_index import FaissIndex, save_mapping
from cir_pipeline.utils.io import ensure_dir, write_json


@torch.no_grad()
def build_gallery_index(
    cfg: AppConfig,
    model,
    gallery_dataset: CIRRDataset,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: torch.device | str,
) -> dict[str, Path]:
    embeddings, names, paths = encode_gallery(
        model,
        gallery_dataset,
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.data.num_workers,
        device=device,
    )
    embeddings_np = embeddings.numpy().astype("float32")
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True).clip(min=1e-12)
    embeddings_np = embeddings_np / norms

    output_dir = ensure_dir(output_dir)
    index = FaissIndex(
        dim=embeddings_np.shape[1],
        index_type=cfg.faiss.index_type,
        hnsw_m=cfg.faiss.hnsw_m,
    )
    index.add(embeddings_np)
    index_path = output_dir / "index.faiss"
    mapping_path = output_dir / "mapping.json"
    metadata_path = output_dir / "metadata.json"
    index.save(index_path)

    mapping = [
        {
            "faiss_id": idx,
            "image_name": name,
            "image_path": path,
            "split": gallery_dataset.split,
        }
        for idx, (name, path) in enumerate(zip(names, paths))
    ]
    save_mapping(mapping, mapping_path)
    write_json(
        {
            "model_checkpoint": str(checkpoint_path),
            "embedding_dim": int(embeddings_np.shape[1]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "normalize": True,
            "index_type": cfg.faiss.index_type,
            "num_gallery_images": len(mapping),
        },
        metadata_path,
    )
    return {"index": index_path, "mapping": mapping_path, "metadata": metadata_path}
