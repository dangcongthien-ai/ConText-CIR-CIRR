"""CLIP encoder wrappers."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import CLIPModel, CLIPTokenizerFast


def _last_hidden_state(outputs) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0]


def _safe_cache_name(repo_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in repo_id)


def _tokenizer_cache_dir(backbone: str) -> Path:
    base_dir = Path(os.environ.get("CIR_PIPELINE_TOKENIZER_CACHE", Path.home() / ".cache" / "cir_pipeline"))
    return base_dir / "tokenizers" / _safe_cache_name(backbone)


def _load_clip_tokenizer(backbone: str) -> CLIPTokenizerFast:
    if Path(backbone).exists():
        return CLIPTokenizerFast.from_pretrained(backbone)
    try:
        return _load_clip_tokenizer_from_hub_files(backbone)
    except Exception as hub_files_error:
        try:
            return CLIPTokenizerFast.from_pretrained(backbone)
        except Exception as pretrained_error:
            raise RuntimeError(
                f"Could not load CLIP tokenizer for {backbone!r}. "
                "If this is a HuggingFace Hub 404 for additional_chat_templates, "
                "upgrade transformers/huggingface_hub or rerun with network access once. "
                f"Direct tokenizer-file loading failed first with: {hub_files_error}"
            ) from pretrained_error


def _load_clip_tokenizer_from_hub_files(backbone: str) -> CLIPTokenizerFast:
    from huggingface_hub import hf_hub_download

    cache_dir = _tokenizer_cache_dir(backbone)
    cache_dir.mkdir(parents=True, exist_ok=True)
    required_files = ("vocab.json", "merges.txt")
    optional_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    for filename in required_files:
        hf_hub_download(repo_id=backbone, filename=filename, local_dir=cache_dir)
    for filename in optional_files:
        try:
            hf_hub_download(repo_id=backbone, filename=filename, local_dir=cache_dir)
        except Exception:
            continue
    try:
        return CLIPTokenizerFast.from_pretrained(str(cache_dir), local_files_only=True)
    except Exception as exc:
        warnings.warn(
            f"Tokenizer files were downloaded to {cache_dir}, but local tokenizer loading failed: {exc}"
        )
        raise


def _load_clip_model(backbone: str) -> CLIPModel:
    try:
        return CLIPModel.from_pretrained(backbone, use_safetensors=True)
    except (OSError, ValueError) as safetensors_error:
        warnings.warn(
            f"Could not load CLIPModel for {backbone!r} with safetensors; "
            f"retrying with PyTorch weights. Original error: {safetensors_error}"
        )
        try:
            return CLIPModel.from_pretrained(backbone, use_safetensors=False)
        except Exception as pytorch_error:
            raise RuntimeError(
                f"Could not load CLIPModel for {backbone!r} with safetensors or PyTorch weights. "
                f"Safetensors error was: {safetensors_error}"
            ) from pytorch_error


class CLIPTokenEncoders(nn.Module):
    """Expose CLIP hidden states used by the ConText-style fusion model."""

    def __init__(self, backbone: str) -> None:
        super().__init__()
        self.clip_model = _load_clip_model(backbone)
        self.tokenizer = _load_clip_tokenizer(backbone)
        self.image_embedding_dim = int(self.vision_model.config.hidden_size)
        self.text_embedding_dim = int(self.text_model.config.hidden_size)
        self.embedding_projection_dim = int(
            getattr(self.clip_model.config, "projection_dim", self.image_embedding_dim)
        )
        for parameter in self.visual_projection.parameters():
            parameter.requires_grad = False
        for parameter in self.text_projection.parameters():
            parameter.requires_grad = False
        self.logit_scale.requires_grad = False

    @property
    def vision_model(self):
        return self.clip_model.vision_model

    @property
    def text_model(self):
        return self.clip_model.text_model

    @property
    def visual_projection(self):
        return self.clip_model.visual_projection

    @property
    def text_projection(self):
        return self.clip_model.text_projection

    @property
    def logit_scale(self):
        return self.clip_model.logit_scale

    def configure_gradient_checkpointing(self, enabled: bool) -> None:
        for model in (self.vision_model, self.text_model):
            if enabled and hasattr(model, "gradient_checkpointing_enable"):
                self._enable_gradient_checkpointing(model)
            elif not enabled and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

    def _enable_gradient_checkpointing(self, model: nn.Module) -> None:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

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
