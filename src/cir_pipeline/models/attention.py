"""Cross-attention fusion and attention pooling modules."""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    """Image tokens attend to text tokens, matching the retrieval branch."""

    def __init__(self, image_dim: int, text_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.text_norm = nn.LayerNorm(text_dim)
        self.image_norm = nn.LayerNorm(image_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=text_dim,
            vdim=text_dim,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, image_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim * 4, image_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.image_norm(image_tokens)
        key_value = self.text_norm(text_tokens)
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()
        attended, weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        image_tokens = image_tokens + attended
        image_tokens = image_tokens + self.ffn(image_tokens)
        return image_tokens, weights


class TextToImageAttentionBlock(nn.Module):
    """Text tokens attend to image tokens, matching the CC-style concept grounding branch."""

    def __init__(self, image_dim: int, text_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.image_norm = nn.LayerNorm(image_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=image_dim,
            vdim=image_dim,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 4, text_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.text_norm(text_tokens)
        key_value = self.image_norm(image_tokens)
        attended, weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=False,
        )
        text_tokens = text_tokens + attended
        text_tokens = text_tokens + self.ffn(text_tokens)
        return text_tokens, weights


class CrossAttentionFusion(nn.Module):
    """Stack retrieval cross-attention layers and expose per-layer attention maps."""

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(image_dim, text_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attention_maps: list[torch.Tensor] = []
        for layer in self.layers:
            image_tokens, weights = layer(image_tokens, text_tokens, text_mask)
            attention_maps.append(weights)
        return image_tokens, attention_maps


class TextToImageAttentionFusion(nn.Module):
    """Stack concept-grounding text-to-image attention layers for Text-CC."""

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TextToImageAttentionBlock(image_dim, text_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attention_maps: list[torch.Tensor] = []
        for layer in self.layers:
            text_tokens, weights = layer(text_tokens, image_tokens)
            attention_maps.append(weights)
        return text_tokens, attention_maps


class AttentionPooling(nn.Module):
    """Set-Transformer-style pooling with a learned seed query."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = tokens.shape[0]
        query = self.seed.expand(batch, -1, -1)
        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask.bool()
        pooled, _ = self.attn(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(pooled.squeeze(1))
