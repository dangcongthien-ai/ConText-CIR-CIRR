"""COMET logger construction and artifact helpers."""

from __future__ import annotations

import inspect
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
    return CometLogger(**_comet_logger_kwargs(CometLogger, cfg, api_key, workspace, project_name))


def _comet_logger_kwargs(
    comet_logger_cls: Any,
    cfg: AppConfig,
    api_key: str | None,
    workspace: str | None,
    project_name: str,
) -> dict[str, Any]:
    parameters = inspect.signature(comet_logger_cls).parameters
    kwargs: dict[str, Any] = {"api_key": api_key, "workspace": workspace}
    if "project" in parameters:
        kwargs["project"] = project_name
    else:
        kwargs["project_name"] = project_name
    if "name" in parameters:
        kwargs["name"] = cfg.logger.experiment_name
    else:
        kwargs["experiment_name"] = cfg.logger.experiment_name
    if "online" in parameters:
        kwargs["online"] = not cfg.logger.offline
    else:
        kwargs["offline"] = cfg.logger.offline
    if "offline_directory" in parameters:
        kwargs["offline_directory"] = cfg.project.output_dir
    else:
        kwargs["save_dir"] = cfg.project.output_dir
    return kwargs


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
