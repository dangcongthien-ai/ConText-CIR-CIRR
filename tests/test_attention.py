from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from cir_pipeline.models.attention import CrossAttentionFusion


def test_cross_attention_uses_image_queries_and_text_keys():
    fusion = CrossAttentionFusion(image_dim=8, text_dim=6, num_layers=1, num_heads=2)
    image_tokens = torch.randn(2, 5, 8)
    text_tokens = torch.randn(2, 7, 6)
    text_mask = torch.ones(2, 7, dtype=torch.bool)

    fused, attention_maps = fusion(image_tokens, text_tokens, text_mask)

    assert fused.shape == (2, 5, 8)
    assert len(attention_maps) == 1
    assert attention_maps[0].shape == (2, 2, 5, 7)
