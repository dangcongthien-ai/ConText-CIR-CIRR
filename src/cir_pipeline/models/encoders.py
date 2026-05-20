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

    def configure_gradient_checkpointing(self, enabled: bool) -> None:
        for model in (self.vision_model, self.text_model):
            if enabled and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif not enabled and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

    def configure_trainable_encoders(
        self,
        freeze_vision_encoder: bool,
        freeze_text_encoder: bool,
        trainable_vision_layers: int = 0,
        trainable_text_layers: int = 0,
    ) -> None:
        self._configure_encoder(self.vision_model, freeze_vision_encoder, trainable_vision_layers)
        self._configure_encoder(self.text_model, freeze_text_encoder, trainable_text_layers)

    def _configure_encoder(self, model: nn.Module, freeze_encoder: bool, trainable_layers: int) -> None:
        for parameter in model.parameters():
            parameter.requires_grad = not freeze_encoder
        if freeze_encoder and trainable_layers > 0:
            self._unfreeze_last_transformer_layers(model, trainable_layers)

    def _unfreeze_last_transformer_layers(self, model: nn.Module, trainable_layers: int) -> None:
        layers = self._encoder_layers(model)
        if not layers:
            raise ValueError(f"Could not find transformer layers on {type(model).__name__}.")
        for layer in layers[-trainable_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for module_name in (
            "final_layer_norm",
            "post_layernorm",
            "vision_model.post_layernorm",
            "text_model.final_layer_norm",
        ):
            module = self._nested_module(model, module_name)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def _encoder_layers(self, model: nn.Module):
        for path in ("vision_model.encoder.layers", "text_model.encoder.layers", "encoder.layers"):
            module = self._nested_module(model, path)
            if module is not None:
                return list(module)
        return []

    def _nested_module(self, model: nn.Module, path: str) -> nn.Module | None:
        current: nn.Module | None = model
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

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
