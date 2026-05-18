"""CLIP encoder wrappers."""

from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPModel, CLIPTokenizerFast


class CLIPTokenEncoders(nn.Module):
    """Expose CLIP image and text token embeddings in the shared CLIP space."""

    def __init__(self, backbone: str) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(backbone, use_safetensors=True)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
        self.embedding_dim = int(self.clip.config.projection_dim)

    def tokenize(self, texts: list[str], device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        return encoded

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.clip.vision_model(pixel_values=images, return_dict=True)
        tokens = outputs.last_hidden_state
        return self.clip.visual_projection(tokens)

    def encode_text_tokens(self, texts: list[str], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenize(texts, device=device)
        outputs = self.clip.text_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
        tokens = self.clip.text_projection(outputs.last_hidden_state)
        return tokens, encoded["attention_mask"].bool()
