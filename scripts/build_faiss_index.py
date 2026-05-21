from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from cir_pipeline.config import load_config
from cir_pipeline.data.cirr_datamodule import CIRRDataModule
from cir_pipeline.engine.comet_logger import build_lightning_logger, log_artifact, log_config
from cir_pipeline.engine.train_module import load_cir_module_from_checkpoint
from cir_pipeline.retrieval.build_gallery import build_gallery_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS gallery index for CIRR.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test1"])
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_cir_module_from_checkpoint(args.checkpoint, cfg=cfg, map_location=device)
    module.model.to(device).eval()
    datamodule = CIRRDataModule(cfg)
    gallery = datamodule.gallery_dataset(args.split)
    output_dir = args.output_dir or Path(cfg.faiss.output_dir) / f"cirr_{args.split}"
    paths = build_gallery_index(cfg, module.model, gallery, args.checkpoint, output_dir, device)
    for key, value in paths.items():
        print(f"{key}: {value}")
    if cfg.logger.log_artifacts:
        logger = build_lightning_logger(cfg)
        log_config(logger, cfg)
        for value in paths.values():
            log_artifact(logger, value)


if __name__ == "__main__":
    main()
