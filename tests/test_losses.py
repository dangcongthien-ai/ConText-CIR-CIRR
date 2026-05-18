from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from cir_pipeline.models.losses import retrieval_contrastive_loss, text_concept_consistency_loss


def test_retrieval_loss_accepts_query_negatives():
    embeddings = F.normalize(torch.eye(4), dim=-1)
    query_neg = F.normalize(torch.flip(torch.eye(4), dims=[0]), dim=-1)
    loss = retrieval_contrastive_loss(
        embeddings,
        embeddings,
        query_negative_embeddings=query_neg,
        temperature=0.07,
        use_query_image_negative=True,
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_text_cc_returns_zero_without_valid_phrases():
    attn = [torch.rand(2, 2, 3, 4)]
    loss = text_concept_consistency_loss(attn, attn, [], [], [], epsilon=0.05)
    assert torch.isclose(loss, loss.new_tensor(0.0)).item()


def test_text_cc_positive_for_different_attention_maps():
    full = [torch.tensor([[[[0.9, 0.1], [0.9, 0.1]]]])]
    iso = [torch.tensor([[[[0.1, 0.9], [0.1, 0.9]]]])]
    loss = text_concept_consistency_loss(
        full,
        iso,
        full_token_indices=[[0]],
        isolated_token_indices=[[0]],
        sample_indices=[0],
        epsilon=0.0,
    )
    assert loss.item() > 0.0
