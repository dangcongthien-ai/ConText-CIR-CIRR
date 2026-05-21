from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from cir_pipeline.config import load_config
from cir_pipeline.data.cirr_datamodule import CIRRDataModule
from cir_pipeline.engine.evaluator import evaluate_retrieval
from cir_pipeline.engine.train_module import load_cir_module_from_checkpoint
from cir_pipeline.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ConText-CIR checkpoint on CIRR.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "train", "test1"])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_cir_module_from_checkpoint(args.checkpoint, cfg=cfg, map_location=device)
    module.model.to(device)
    datamodule = CIRRDataModule(cfg)
    relative = datamodule.relative_dataset(args.split)
    gallery = datamodule.gallery_dataset(args.split)
    result = evaluate_retrieval(
        module.model,
        relative,
        gallery,
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.data.num_workers,
        device=device,
        top_k=cfg.faiss.top_k,
    )
    print(json.dumps(result.metrics, indent=2))
    if args.output:
        write_json({"metrics": result.metrics, "predictions": result.predictions}, args.output)


if __name__ == "__main__":
    main()
