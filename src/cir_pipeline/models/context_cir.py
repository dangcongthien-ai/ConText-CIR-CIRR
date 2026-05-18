"""Main ConText-CIR-like model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cir_pipeline.config import ModelConfig
from cir_pipeline.models.attention import AttentionPooling, CrossAttentionFusion
from cir_pipeline.models.encoders import CLIPTokenEncoders


class ConTextCIR(nn.Module):
    """Fuse image tokens and modification text tokens into retrieval embeddings."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoders = CLIPTokenEncoders(cfg.backbone)
        dim = self.encoders.embedding_dim
        self.fusion = CrossAttentionFusion(
            dim=dim,
            num_layers=cfg.num_cross_attention_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.pooling = AttentionPooling(dim=dim, num_heads=cfg.num_heads, dropout=cfg.dropout)
        output_dim = cfg.projection_dim or dim
        self.projection = nn.Identity() if output_dim == dim else nn.Linear(dim, output_dim)
        self.embedding_dim = output_dim

    @property
    def tokenizer(self):
        return self.encoders.tokenizer

    def forward_query(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        return self.forward_with_attention(images, texts)["embeddings"]

    def encode_gallery(self, images: torch.Tensor) -> torch.Tensor:
        """Encode gallery images as f(image, ''), matching the query space."""

        blank_texts = [""] * images.shape[0]
        return self.forward_query(images, blank_texts)

    def forward_with_attention(
        self,
        images: torch.Tensor,
        texts: list[str],
        token_indices: list[list[int]] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        del token_indices
        image_tokens = self.encoders.encode_image_tokens(images)
        text_tokens, text_mask = self.encoders.encode_text_tokens(texts, device=images.device)
        fused_tokens, attention_maps = self.fusion(text_tokens, image_tokens)
        pooled = self.pooling(fused_tokens, text_mask)
        embeddings = self.projection(pooled)
        if self.cfg.normalize_embeddings:
            embeddings = F.normalize(embeddings, dim=-1)
        return {
            "embeddings": embeddings,
            "attention_maps": attention_maps,
            "text_attention_mask": text_mask,
        }
