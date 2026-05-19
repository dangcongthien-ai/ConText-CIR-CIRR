"""CLIP encoder wrappers."""

from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizerFast, CLIPVisionModel


def _last_hidden_state(outputs) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0]


class CLIPTokenEncoders(nn.Module):
    """Expose CLIP hidden states used by the ConText-style fusion model."""

    def __init__(self, backbone: str) -> None:
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(backbone, use_safetensors=True)
        self.text_model = CLIPTextModel.from_pretrained(backbone, use_safetensors=True)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(backbone)
        self.image_embedding_dim = int(self.vision_model.config.hidden_size)
        self.text_embedding_dim = int(self.text_model.config.hidden_size)

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
        outputs = self.vision_model(pixel_values=images)
        return _last_hidden_state(outputs)

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=images)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return _last_hidden_state(outputs)[:, 0]

    def encode_text_tokens(self, texts: list[str], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenize(texts, device=device)
        outputs = self.text_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        return _last_hidden_state(outputs), encoded["attention_mask"].bool()
