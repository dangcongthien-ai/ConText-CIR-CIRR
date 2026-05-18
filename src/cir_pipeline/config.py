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
    num_workers: int = 8


@dataclass
class ModelConfig:
    backbone: str = "openai/clip-vit-base-patch32"
    backbone_size: str = "B"
    num_cross_attention_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    projection_dim: Optional[int] = None
    normalize_embeddings: bool = True


@dataclass
class LossConfig:
    temperature: float = 0.07
    lambda_cc: float = 0.08
    epsilon_cc: float = 0.05
    max_noun_phrases: int = 10
    use_query_image_negative: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 128
    val_batch_size: int = 256
    max_epochs: int = 30
    learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-2
    warmup_ratio: float = 0.05
    precision: str = "16-mixed"
    accelerator: str = "gpu"
    devices: int | str = 1
    gradient_clip_val: float = 1.0
    val_check_interval: float = 1.0
    save_top_k: int = 3
    monitor: str = "val/recall@10"
    monitor_mode: str = "max"


@dataclass
class LoggerConfig:
    name: str = "comet"
    offline: bool = False
    project_name: str = "ConText-CIR-CIRR"
    workspace: Optional[str] = None
    experiment_name: str = "context_cir_cirr_baseline"
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
    if cfg.data.dataset != "cirr":
        raise ValueError("Only data.dataset='cirr' is supported.")
    if cfg.loss.temperature <= 0:
        raise ValueError("loss.temperature must be > 0.")
    if cfg.loss.lambda_cc < 0:
        raise ValueError("loss.lambda_cc must be >= 0.")
    if cfg.model.num_cross_attention_layers < 1:
        raise ValueError("model.num_cross_attention_layers must be >= 1.")
    if cfg.model.num_heads < 1:
        raise ValueError("model.num_heads must be >= 1.")
    if cfg.faiss.index_type not in {"flat_ip", "hnsw"}:
        raise ValueError("faiss.index_type must be 'flat_ip' or 'hnsw'.")


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: AppConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(cfg), handle, sort_keys=False)
