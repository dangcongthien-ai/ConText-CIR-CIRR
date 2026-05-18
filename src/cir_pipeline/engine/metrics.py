"""CIRR retrieval metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def rank_names(
    scores: np.ndarray,
    gallery_names: Sequence[str],
    reference_name: str | None = None,
) -> list[str]:
    order = np.argsort(-scores)
    ranked = [gallery_names[idx] for idx in order]
    if reference_name is not None:
        ranked = [name for name in ranked if name != reference_name]
    return ranked


def recall_at_k(
    ranked_names: Sequence[Sequence[str]],
    target_names: Sequence[str | None],
    ks: Sequence[int] = (1, 5, 10, 50),
) -> dict[str, float]:
    valid = [(ranked, target) for ranked, target in zip(ranked_names, target_names) if target]
    if not valid:
        return {f"recall@{k}": 0.0 for k in ks}
    metrics: dict[str, float] = {}
    for k in ks:
        hits = sum(1 for ranked, target in valid if target in ranked[:k])
        metrics[f"recall@{k}"] = hits / len(valid)
    return metrics


def recall_subset_at_k(
    ranked_names: Sequence[Sequence[str]],
    target_names: Sequence[str | None],
    img_sets: Sequence[Sequence[str]],
    ks: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    valid = []
    for ranked, target, img_set in zip(ranked_names, target_names, img_sets):
        if target and img_set:
            subset_ranked = [name for name in ranked if name in set(img_set)]
            valid.append((subset_ranked, target))
    if not valid:
        return {f"recall_subset@{k}": 0.0 for k in ks}
    metrics: dict[str, float] = {}
    for k in ks:
        hits = sum(1 for ranked, target in valid if target in ranked[:k])
        metrics[f"recall_subset@{k}"] = hits / len(valid)
    return metrics
