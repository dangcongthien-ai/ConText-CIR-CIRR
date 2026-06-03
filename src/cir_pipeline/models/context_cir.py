"""Main ConText-CIR-like model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cir_pipeline.config import ModelConfig
from cir_pipeline.models.attention import AttentionPooling, CrossAttentionFusion, TextToImageAttentionFusion
from cir_pipeline.models.encoders import CLIPTokenEncoders


class ConTextCIR(nn.Module):
    """Fuse image tokens and modification text tokens into retrieval embeddings."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoders = CLIPTokenEncoders(cfg.backbone)
        self.encoders.configure_gradient_checkpointing(cfg.use_gradient_checkpointing)
        self.encoders.configure_trainable_encoders(
            freeze_vision_encoder=cfg.freeze_vision_encoder,
            freeze_text_encoder=cfg.freeze_text_encoder,
            trainable_vision_layers=cfg.trainable_vision_layers,
            trainable_text_layers=cfg.trainable_text_layers,
        )
        image_dim = self.encoders.image_embedding_dim
        text_dim = self.encoders.text_embedding_dim
        self.fusion = CrossAttentionFusion(
            image_dim=image_dim,
            text_dim=text_dim,
            num_layers=cfg.num_cross_attention_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.cc_fusion = TextToImageAttentionFusion(
            image_dim=image_dim,
            text_dim=text_dim,
            num_layers=cfg.num_cross_attention_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.pooling = AttentionPooling(dim=image_dim, num_heads=cfg.num_heads, dropout=cfg.dropout)
        native_projection_dim = self.encoders.embedding_projection_dim
        output_dim = cfg.projection_dim or (native_projection_dim if cfg.use_native_clip_projection else image_dim)
        self.projection = self._build_projection(image_dim, output_dim, cfg.use_native_clip_projection)
        self.embedding_dim = output_dim

    @property
    def tokenizer(self):
        return self.encoders.tokenizer

    def _build_projection(self, image_dim: int, output_dim: int, use_native_clip_projection: bool) -> nn.Module:
        if output_dim == image_dim and not use_native_clip_projection:
            return nn.Identity()
        projection = nn.Linear(image_dim, output_dim, bias=False)
        if use_native_clip_projection:
            native_projection = self.encoders.visual_projection
            if native_projection is not None and native_projection.weight.shape == projection.weight.shape:
                with torch.no_grad():
                    projection.weight.copy_(native_projection.weight)
        return projection

    def _encode_query_inputs(
        self,
        images: torch.Tensor,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_tokens = self.encoders.encode_image_tokens(images)
        text_tokens, text_mask = self.encoders.encode_text_tokens(texts, device=images.device)
        return image_tokens, text_tokens, text_mask

    def forward_query(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        return self.forward_with_attention(images, texts)["embeddings"]

    def encode_gallery(self, images: torch.Tensor) -> torch.Tensor:
        """Encode gallery/target images with the pure CLIP vision branch."""

        embeddings = self.projection(self.encoders.encode_image_features(images))
        if self.cfg.normalize_embeddings:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def forward_with_attention(
        self,
        images: torch.Tensor,
        texts: list[str],
        token_indices: list[list[int]] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | str]:
        del token_indices
        image_tokens, text_tokens, text_mask = self._encode_query_inputs(images, texts)
        fused_tokens, attention_maps = self.fusion(image_tokens, text_tokens, text_mask)
        pooled = self.pooling(fused_tokens)
        embeddings = self.projection(pooled)
        if self.cfg.normalize_embeddings:
            embeddings = F.normalize(embeddings, dim=-1)
        return {
            "embeddings": embeddings,
            "attention_maps": attention_maps,
            "text_attention_mask": text_mask,
            "attention_layout": "image_to_text",
        }

    def forward_text_cc_attention(
        self,
        images: torch.Tensor,
        texts: list[str],
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | str]:
        image_tokens, text_tokens, text_mask = self._encode_query_inputs(images, texts)
        fused_tokens, attention_maps = self.cc_fusion(text_tokens, image_tokens)
        return {
            "tokens": fused_tokens,
            "attention_maps": attention_maps,
            "text_attention_mask": text_mask,
            "attention_layout": "text_to_image",
        }
