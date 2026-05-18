"""Full-gallery CIRR validation and prediction loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cir_pipeline.data.cirr_dataset import CIRRDataset
from cir_pipeline.data.collate import cirr_collate
from cir_pipeline.engine.metrics import rank_names, recall_at_k, recall_subset_at_k


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    predictions: dict[str, list[str]]
    subset_predictions: dict[str, list[str]]
    gallery_names: list[str]


@torch.no_grad()
def encode_gallery(
    model,
    dataset: CIRRDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, list[str], list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=cirr_collate,
    )
    embeddings = []
    names: list[str] = []
    paths: list[str] = []
    model.eval()
    for batch in tqdm(loader, desc=f"Encode {dataset.split} gallery", leave=False):
        image = batch["image"].to(device)
        emb = model.encode_gallery(image)
        embeddings.append(emb.detach().cpu())
        names.extend(batch["image_name"])
        paths.extend(batch["image_path"])
    return torch.cat(embeddings, dim=0), names, paths


@torch.no_grad()
def encode_queries(
    model,
    dataset: CIRRDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, list[Any]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=cirr_collate,
    )
    embeddings = []
    meta: dict[str, list[Any]] = {
        "pair_id": [],
        "reference_name": [],
        "target_name": [],
        "img_set": [],
    }
    model.eval()
    for batch in tqdm(loader, desc=f"Encode {dataset.split} queries", leave=False):
        ref = batch["reference_image"].to(device)
        emb = model.forward_query(ref, batch["caption"])
        embeddings.append(emb.detach().cpu())
        for key in meta:
            meta[key].extend(batch[key])
    return torch.cat(embeddings, dim=0), meta


@torch.no_grad()
def evaluate_retrieval(
    model,
    relative_dataset: CIRRDataset,
    gallery_dataset: CIRRDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device | str,
    top_k: int = 50,
) -> EvaluationResult:
    gallery_emb, gallery_names, _ = encode_gallery(
        model, gallery_dataset, batch_size, num_workers, device
    )
    query_emb, meta = encode_queries(model, relative_dataset, batch_size, num_workers, device)
    scores = (query_emb @ gallery_emb.t()).numpy()

    ranked_all: list[list[str]] = []
    predictions: dict[str, list[str]] = {}
    subset_predictions: dict[str, list[str]] = {}
    for idx, row_scores in enumerate(scores):
        ranked = rank_names(row_scores, gallery_names, meta["reference_name"][idx])
        ranked_all.append(ranked)
        pair_id = str(int(meta["pair_id"][idx]))
        predictions[pair_id] = ranked[:top_k]
        img_set = meta["img_set"][idx]
        if img_set:
            subset_ranked = [name for name in ranked if name in set(img_set)]
            subset_predictions[pair_id] = subset_ranked[:3]

    metrics = {}
    metrics.update(recall_at_k(ranked_all, meta["target_name"], ks=(1, 5, 10, 50)))
    metrics.update(
        recall_subset_at_k(ranked_all, meta["target_name"], meta["img_set"], ks=(1, 2, 3))
    )
    metrics["num_gallery_images"] = float(len(gallery_names))
    metrics["num_queries"] = float(len(ranked_all))
    return EvaluationResult(metrics, predictions, subset_predictions, gallery_names)
