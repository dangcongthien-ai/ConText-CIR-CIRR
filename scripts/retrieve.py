from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from cir_pipeline.config import load_config
from cir_pipeline.engine.comet_logger import build_lightning_logger, log_artifact, log_config
from cir_pipeline.engine.train_module import load_cir_module_from_checkpoint
from cir_pipeline.retrieval.search import encode_query_image_text, save_retrieval_results, search_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve CIRR images with image + text query.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--query_image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--reference_name", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_cir_module_from_checkpoint(args.checkpoint, cfg=cfg, map_location=device)
    module.model.to(device).eval()

    query_embedding = encode_query_image_text(module.model, args.query_image, args.text, device)
    reference_name = args.reference_name or Path(args.query_image).stem
    results = search_index(
        args.index,
        args.mapping,
        query_embedding,
        top_k=args.top_k,
        reference_image_name=reference_name,
    )
    print(f"{'rank':>4}  {'score':>9}  image_name")
    for row in results:
        print(f"{row['rank']:>4}  {row['score']:>9.4f}  {row['image_name']}")

    output = args.output
    if output is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(cfg.project.output_dir) / "retrieval_runs" / run_id / "results.json"
    save_retrieval_results(results, output)
    if cfg.logger.log_artifacts:
        logger = build_lightning_logger(cfg)
        log_config(logger, cfg)
        log_artifact(logger, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
