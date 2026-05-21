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
from cir_pipeline.engine.evaluator import evaluate_retrieval
from cir_pipeline.engine.train_module import load_cir_module_from_checkpoint
from cir_pipeline.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CIRR test1 submission JSON files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--submission_name", default="context_cir_cirr")
    parser.add_argument("--output_dir", default="outputs/submissions/cirr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_cir_module_from_checkpoint(args.checkpoint, cfg=cfg, map_location=device)
    module.model.to(device).eval()
    datamodule = CIRRDataModule(cfg)
    relative = datamodule.relative_dataset("test1")
    gallery = datamodule.gallery_dataset("test1")
    result = evaluate_retrieval(
        module.model,
        relative,
        gallery,
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.data.num_workers,
        device=device,
        top_k=50,
    )
    output_dir = ensure_dir(args.output_dir)
    submission = {"version": "rc2", "metric": "recall"}
    subset_submission = {"version": "rc2", "metric": "recall_subset"}
    submission.update(result.predictions)
    subset_submission.update(result.subset_predictions)
    main_path = output_dir / f"{args.submission_name}.json"
    subset_path = output_dir / f"subset_{args.submission_name}.json"
    write_json(submission, main_path)
    write_json(subset_submission, subset_path)
    if cfg.logger.log_artifacts:
        logger = build_lightning_logger(cfg)
        log_config(logger, cfg)
        log_artifact(logger, main_path)
        log_artifact(logger, subset_path)
    print(f"saved: {main_path}")
    print(f"saved: {subset_path}")


if __name__ == "__main__":
    main()
