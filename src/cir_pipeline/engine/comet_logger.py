"""COMET logger construction and artifact helpers."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from cir_pipeline.config import AppConfig, config_to_dict
from cir_pipeline.utils.env import load_project_env


def build_lightning_logger(cfg: AppConfig):
    """Create a Lightning logger, falling back safely when unavailable."""

    load_project_env()
    name = (cfg.logger.name or "none").lower()
    if name == "none":
        return False
    if name == "tensorboard":
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"TensorBoard logger unavailable: {exc}")
            return False
        return TensorBoardLogger(save_dir=cfg.project.output_dir, name=cfg.project.name)
    if name != "comet":
        warnings.warn(f"Unknown logger '{cfg.logger.name}', disabling logger.")
        return False

    api_key = os.getenv("COMET_API_KEY")
    workspace = cfg.logger.workspace or os.getenv("COMET_WORKSPACE")
    project_name = os.getenv("COMET_PROJECT_NAME") or cfg.logger.project_name
    try:
        from pytorch_lightning.loggers import CometLogger
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"COMET logger unavailable: {exc}")
        return False

    if not api_key and not cfg.logger.offline:
        warnings.warn(
            "logger.name=comet but COMET_API_KEY is missing; training will continue without COMET."
        )
        return False
    return CometLogger(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        save_dir=cfg.project.output_dir,
        offline=cfg.logger.offline,
        experiment_name=cfg.logger.experiment_name,
    )


def log_config(logger: Any, cfg: AppConfig) -> None:
    if not logger or not getattr(logger, "experiment", None):
        return
    experiment = logger.experiment
    if hasattr(experiment, "log_parameters"):
        experiment.log_parameters(config_to_dict(cfg))


def log_artifact(logger: Any, path: str | Path, name: str | None = None) -> None:
    if not logger or not getattr(logger, "experiment", None):
        return
    path = Path(path)
    if not path.exists():
        return
    experiment = logger.experiment
    if hasattr(experiment, "log_asset"):
        experiment.log_asset(str(path), file_name=name or path.name)
