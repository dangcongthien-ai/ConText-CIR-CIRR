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
    query_negative_weight: float = 1.0,
) -> torch.Tensor:
    """Symmetric InfoNCE with an optional weighted reference-image penalty."""

    query_to_target = query_embeddings @ target_embeddings.t()
    query_to_target = query_to_target / temperature
    target_to_query = (target_embeddings @ query_embeddings.t()) / temperature
    labels = torch.arange(query_embeddings.shape[0], device=query_embeddings.device)
    loss_query = F.cross_entropy(query_to_target, labels)
    loss_target = F.cross_entropy(target_to_query, labels)
    loss = 0.5 * (loss_query + loss_target)
    if use_query_image_negative and query_negative_embeddings is not None and query_negative_weight > 0:
        positive_logits = (query_embeddings * target_embeddings).sum(dim=-1) / temperature
        reference_logits = (query_embeddings * query_negative_embeddings).sum(dim=-1) / temperature
        query_negative_loss = F.softplus(reference_logits - positive_logits).mean()
        loss = loss + query_negative_weight * query_negative_loss
    return loss


def _average_attention(attention_maps: list[torch.Tensor]) -> torch.Tensor:
    if not attention_maps:
        raise ValueError("Expected at least one attention map for Text CC loss.")
    stacked = torch.stack(attention_maps, dim=0)
    if stacked.ndim != 5:
        raise ValueError("Expected attention maps with shape (batch, heads, image_tokens, text_tokens).")
    return stacked.mean(dim=(0, 2))


def text_concept_consistency_loss(
    full_attention_maps: list[torch.Tensor],
    isolated_attention_maps: list[torch.Tensor],
    full_token_indices: list[list[int]],
    isolated_token_indices: list[list[int]],
    sample_indices: list[int],
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Compare full-caption and isolated noun-phrase image-token attention distributions."""

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
        valid_full = full_idx_tensor[full_idx_tensor < full_attn.shape[-1]]
        valid_iso = iso_idx_tensor[iso_idx_tensor < iso_attn.shape[-1]]
        if valid_full.numel() == 0 or valid_iso.numel() == 0:
            continue
        full_dist = _phrase_image_distribution(full_attn[sample_idx], valid_full)
        iso_dist = _phrase_image_distribution(iso_attn[iso_idx], valid_iso)
        losses.append(torch.relu(_jensen_shannon_divergence(full_dist, iso_dist) - epsilon))

    if not losses:
        return full_attn.new_zeros(())
    return torch.stack(losses).mean()


def _phrase_image_distribution(attention: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    phrase_attention = attention[:, token_indices].mean(dim=-1)
    if phrase_attention.shape[0] > 1:
        phrase_attention = phrase_attention[1:]
    phrase_attention = phrase_attention.clamp_min(0)
    return phrase_attention / phrase_attention.sum().clamp_min(1e-8)


def _jensen_shannon_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.clamp_min(1e-8)
    second = second.clamp_min(1e-8)
    mixture = 0.5 * (first + second)
    return 0.5 * (
        F.kl_div(mixture.log(), first, reduction="sum")
        + F.kl_div(mixture.log(), second, reduction="sum")
    )
