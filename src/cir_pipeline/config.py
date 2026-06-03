"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass
class ProjectConfig:
    name: str = "CIR_Pipeline"
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class DataConfig:
    dataset: str = "cirr"
    root: str = "CIRR"
    image_root_train: str = "CIRR/train"
    image_root_val: str = "CIRR/dev"
    image_root_test1: str = "CIRR/test1"
    metadata_root: str = "CIRR/cirr"
    use_extended_captions: bool = False
    skip_empty_captions: bool = True
    noun_phrase_backend: str = "regex"
    noun_phrase_cache_dir: Optional[str] = "noun_phrases"
    use_exported_noun_phrases: bool = True
    stanza_resources_dir: Optional[str] = None
    stanza_download_method: str = "reuse_resources"
    num_workers: int = 8
    noun_phrase_extractor: str = "constituency"
    image_transform: str = "clip"
    target_pad_ratio: float = 1.25


@dataclass
class ModelConfig:
    backbone: str = "openai/clip-vit-base-patch32"
    backbone_size: str = "B"
    num_cross_attention_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    projection_dim: Optional[int] = None
    use_native_clip_projection: bool = False
    normalize_embeddings: bool = True
    freeze_vision_encoder: bool = True
    freeze_text_encoder: bool = True
    trainable_vision_layers: int = 0
    trainable_text_layers: int = 0
    use_gradient_checkpointing: bool = False


@dataclass
class LossConfig:
    temperature: float = 0.07
    lambda_cont: float = 1.0
    lambda_cc: float = 0.08
    epsilon_cc: float = 0.05
    max_noun_phrases: int = 10
    use_query_image_negative: bool = True
    query_negative_weight: float = 1.0
    text_cc_clean_only: bool = False
    text_cc_clean_min_confidence: float = 0.7
    text_cc_attention_direction: str = "text_to_image"


@dataclass
class TrainConfig:
    batch_size: int = 128
    val_batch_size: int = 256
    max_epochs: int = 30
    learning_rate: float = 1.0e-5
    encoder_learning_rate: Optional[float] = None
    weight_decay: float = 1.0e-2
    encoder_weight_decay: Optional[float] = None
    warmup_ratio: float = 0.05
    precision: str = "16-mixed"
    accelerator: str = "gpu"
    devices: int | str = 1
    gradient_clip_val: float = 1.0
    val_check_interval: float = 1.0
    accumulate_grad_batches: int = 1
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    save_top_k: int = 1
    monitor: str = "val/recall@10"
    monitor_mode: str = "max"
    log_cc_grad_norm: bool = False
    cc_grad_norm_interval: int = 0


@dataclass
class LoggerConfig:
    name: str = "comet"
    offline: bool = False
    project_name: str = "ConText-CIR-CIRR"
    workspace: Optional[str] = None
    experiment_name: str = "cirr_vitb32_cl_cc"
    log_artifacts: bool = True


@dataclass
class FaissConfig:
    index_type: str = "flat_ip"
    top_k: int = 50
    output_dir: str = "outputs/faiss"
    hnsw_m: int = 32
    hnsw_ef_search: int = 64


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)


T = TypeVar("T")


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _from_dict(cls: Type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    defaults = cls()
    for field in fields(cls):
        value = data.get(field.name, getattr(defaults, field.name))
        field_type = _strip_optional(type_hints.get(field.name, field.type))
        if is_dataclass(field_type) and isinstance(value, dict):
            value = _from_dict(field_type, value)
        kwargs[field.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML config into typed dataclasses."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = _from_dict(AppConfig, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    _validate_data_config(cfg.data)
    _validate_loss_config(cfg.loss)
    _validate_model_config(cfg.model)
    _validate_train_config(cfg.train)
    _validate_faiss_config(cfg.faiss)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_data_config(cfg: DataConfig) -> None:
    _require(cfg.dataset == "cirr", "Only data.dataset='cirr' is supported.")
    _require(
        cfg.noun_phrase_backend in {"auto", "stanza", "nltk", "regex", "none"},
        "data.noun_phrase_backend must be one of: auto, stanza, nltk, regex, none.",
    )
    _require(
        cfg.noun_phrase_extractor in {"constituency", "dependency", "constituency_minimal_np"},
        "data.noun_phrase_extractor must be one of: constituency, dependency, constituency_minimal_np.",
    )
    _require(
        cfg.stanza_download_method in {"none", "reuse_resources", "download_resources"},
        "data.stanza_download_method must be one of: none, reuse_resources, download_resources.",
    )
    _require(
        cfg.image_transform in {"clip", "squarepad", "targetpad"},
        "data.image_transform must be one of: clip, squarepad, targetpad.",
    )
    _require(cfg.target_pad_ratio > 0, "data.target_pad_ratio must be > 0.")


def _validate_loss_config(cfg: LossConfig) -> None:
    _require(cfg.temperature > 0, "loss.temperature must be > 0.")
    _require(cfg.lambda_cont >= 0, "loss.lambda_cont must be >= 0.")
    _require(cfg.lambda_cc >= 0, "loss.lambda_cc must be >= 0.")
    _require(cfg.lambda_cont > 0 or cfg.lambda_cc > 0, "At least one loss weight must be > 0.")
    _require(cfg.query_negative_weight >= 0, "loss.query_negative_weight must be >= 0.")
    _require(0 <= cfg.text_cc_clean_min_confidence <= 1, "loss.text_cc_clean_min_confidence must be within [0, 1].")
    _require(
        cfg.text_cc_attention_direction in {"image_to_text", "text_to_image"},
        "loss.text_cc_attention_direction must be one of: image_to_text, text_to_image.",
    )


def _validate_model_config(cfg: ModelConfig) -> None:
    _require(cfg.num_cross_attention_layers >= 1, "model.num_cross_attention_layers must be >= 1.")
    _require(cfg.num_heads >= 1, "model.num_heads must be >= 1.")
    _require(cfg.trainable_vision_layers >= 0, "model.trainable_vision_layers must be >= 0.")
    _require(cfg.trainable_text_layers >= 0, "model.trainable_text_layers must be >= 0.")


def _validate_train_config(cfg: TrainConfig) -> None:
    _require(cfg.learning_rate > 0, "train.learning_rate must be > 0.")
    _require(
        cfg.encoder_learning_rate is None or cfg.encoder_learning_rate > 0,
        "train.encoder_learning_rate must be > 0 when set.",
    )
    _require(cfg.weight_decay >= 0, "train.weight_decay must be >= 0.")
    _require(
        cfg.encoder_weight_decay is None or cfg.encoder_weight_decay >= 0,
        "train.encoder_weight_decay must be >= 0 when set.",
    )
    _require(cfg.accumulate_grad_batches >= 1, "train.accumulate_grad_batches must be >= 1.")
    _require(cfg.early_stopping_patience >= 0, "train.early_stopping_patience must be >= 0.")
    _require(cfg.early_stopping_min_delta >= 0, "train.early_stopping_min_delta must be >= 0.")
    _require(cfg.cc_grad_norm_interval >= 0, "train.cc_grad_norm_interval must be >= 0.")


def _validate_faiss_config(cfg: FaissConfig) -> None:
    _require(cfg.index_type in {"flat_ip", "hnsw"}, "faiss.index_type must be 'flat_ip' or 'hnsw'.")


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: AppConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(cfg), handle, sort_keys=False)
