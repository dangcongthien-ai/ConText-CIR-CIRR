"""Losses for ConText-CIR-like training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def retrieval_contrastive_loss(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    query_negative_embeddings: torch.Tensor | None = None,
    temperature: float = 0.07,
    use_query_image_negative: bool = True,
) -> torch.Tensor:
    """InfoNCE over in-batch targets plus optional query-image negatives."""

    logits = query_embeddings @ target_embeddings.t()
    if use_query_image_negative and query_negative_embeddings is not None:
        neg_logits = query_embeddings @ query_negative_embeddings.t()
        logits = torch.cat([logits, neg_logits], dim=1)
    logits = logits / temperature
    labels = torch.arange(query_embeddings.shape[0], device=query_embeddings.device)
    return F.cross_entropy(logits, labels)


def _average_attention(attention_maps: list[torch.Tensor]) -> torch.Tensor:
    if not attention_maps:
        raise ValueError("Expected at least one attention map for Text CC loss.")
    stacked = torch.stack(attention_maps, dim=0)
    return stacked.mean(dim=(0, 2))


def text_concept_consistency_loss(
    full_attention_maps: list[torch.Tensor],
    isolated_attention_maps: list[torch.Tensor],
    full_token_indices: list[list[int]],
    isolated_token_indices: list[list[int]],
    sample_indices: list[int],
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Compare full-caption and isolated noun-phrase image attention maps."""

    if not full_token_indices or not isolated_token_indices or not sample_indices:
        ref = full_attention_maps[0]
        return ref.new_zeros(())

    full_attn = _average_attention(full_attention_maps)
    iso_attn = _average_attention(isolated_attention_maps)
    losses = []
    for iso_idx, sample_idx in enumerate(sample_indices):
        full_idx = full_token_indices[iso_idx]
        iso_token_idx = isolated_token_indices[iso_idx]
        if not full_idx or not iso_token_idx:
            continue
        full_idx_tensor = torch.tensor(full_idx, device=full_attn.device)
        iso_idx_tensor = torch.tensor(iso_token_idx, device=iso_attn.device)
        valid_full = full_idx_tensor[full_idx_tensor < full_attn.shape[1]]
        valid_iso = iso_idx_tensor[iso_idx_tensor < iso_attn.shape[1]]
        if valid_full.numel() == 0 or valid_iso.numel() == 0:
            continue
        a_full = full_attn[sample_idx, valid_full].mean(dim=0)
        a_iso = iso_attn[iso_idx, valid_iso].mean(dim=0)
        losses.append(torch.relu(torch.abs(a_full - a_iso) - epsilon).mean())

    if not losses:
        return full_attn.new_zeros(())
    return torch.stack(losses).mean()
