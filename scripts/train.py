from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cir_pipeline.config import load_config, save_config
from cir_pipeline.data.cirr_datamodule import CIRRDataModule
from cir_pipeline.engine.comet_logger import build_lightning_logger, log_artifact, log_config
from cir_pipeline.engine.train_module import CIRLightningModule
from cir_pipeline.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConText-CIR-like model on CIRR.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume.")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run one train/val batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    accelerator = cfg.train.accelerator
    devices = cfg.train.devices
    precision = cfg.train.precision
    if accelerator == "gpu" and not torch.cuda.is_available():
        print("WARNING: GPU requested but CUDA is unavailable; falling back to CPU.")
        accelerator = "cpu"
        devices = 1
        precision = "32-true"

    datamodule = CIRRDataModule(cfg)
    module = CIRLightningModule(cfg)
    logger = build_lightning_logger(cfg)
    log_config(logger, cfg)

    checkpoint_dir = ROOT / "checkpoints" / cfg.logger.experiment_name
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:03d}",
            monitor=cfg.train.monitor,
            mode=cfg.train.monitor_mode,
            save_top_k=cfg.train.save_top_k,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        val_check_interval=cfg.train.val_check_interval,
        default_root_dir=cfg.project.output_dir,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)

    if cfg.logger.log_artifacts:
        config_copy = Path(cfg.project.output_dir) / "last_config.yaml"
        save_config(cfg, config_copy)
        log_artifact(logger, config_copy)
        best_path = getattr(callbacks[0], "best_model_path", "")
        if best_path:
            log_artifact(logger, best_path)


if __name__ == "__main__":
    main()
