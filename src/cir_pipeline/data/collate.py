"""Batch collation for CIRR samples."""

from __future__ import annotations

from typing import Any

import torch


def _maybe_stack(items: list[Any]) -> torch.Tensor | None:
    if not items or any(item is None for item in items):
        return None
    if torch.is_tensor(items[0]):
        return torch.stack(items, dim=0)
    return None


def cirr_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate relative-query and gallery CIRR samples."""

    out: dict[str, Any] = {}
    keys = set().union(*(sample.keys() for sample in batch))
    for key in keys:
        values = [sample.get(key) for sample in batch]
        if key in {"reference_image", "target_image", "image"}:
            out[key] = _maybe_stack(values)
        else:
            out[key] = values
    return out
