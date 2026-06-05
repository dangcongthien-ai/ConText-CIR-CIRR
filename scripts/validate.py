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
from cir_pipeline.engine.evaluator import (
    encode_gallery,
    encode_queries,
    evaluate_retrieval,
    evaluate_retrieval_from_embeddings,
)
from cir_pipeline.engine.train_module import load_cir_module_from_checkpoint
from cir_pipeline.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ConText-CIR checkpoint on CIRR.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "train", "test1"])
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--reference_penalty_weight",
        type=float,
        default=0.0,
        help="Penalty weight applied to candidates that are too similar to the reference image.",
    )
    parser.add_argument(
        "--reference_penalty_top_k",
        type=int,
        default=50,
        help="Only rerank the first-stage top-K candidates for each query. Use 0 to rerank the full gallery.",
    )
    parser.add_argument(
        "--reference_penalty_grid",
        default=None,
        help="Optional comma-separated list of reference-penalty weights to scan on the validation split.",
    )
    return parser.parse_args()


def _parse_grid(argument: str | None) -> list[float]:
    if not argument:
        return []
    values: list[float] = []
    for raw_value in argument.split(","):
        stripped = raw_value.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


def _grid_search(
    model,
    relative,
    gallery,
    cfg,
    device,
    weights: list[float],
    reference_penalty_top_k: int,
) -> tuple[dict[str, float], dict[str, list[str]], list[dict[str, float]]]:
    gallery_emb, gallery_names, _ = encode_gallery(
        model,
        gallery,
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.data.num_workers,
        device=device,
    )
    query_emb, meta = encode_queries(
        model,
        relative,
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.data.num_workers,
        device=device,
    )

    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] | None = None
    best_predictions: dict[str, list[str]] | None = None
    best_score = float("-inf")

    for weight in weights:
        result = evaluate_retrieval_from_embeddings(
            query_emb,
            gallery_emb,
            gallery_names,
            meta,
            top_k=cfg.faiss.top_k,
            reference_penalty_weight=weight,
            reference_penalty_top_k=reference_penalty_top_k,
        )
        metrics = dict(result.metrics)
        history.append(metrics)
        score = float(metrics.get("recall@10", 0.0))
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_predictions = result.predictions

    if best_metrics is None or best_predictions is None:
        raise RuntimeError("Reference-penalty grid search produced no results.")
    return best_metrics, best_predictions, history


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_cir_module_from_checkpoint(args.checkpoint, cfg=cfg, map_location=device)
    module.model.to(device)
    datamodule = CIRRDataModule(cfg)
    relative = datamodule.relative_dataset(args.split)
    gallery = datamodule.gallery_dataset(args.split)

    grid_weights = _parse_grid(args.reference_penalty_grid)
    if grid_weights:
        metrics, predictions, history = _grid_search(
            module.model,
            relative,
            gallery,
            cfg,
            device,
            grid_weights,
            args.reference_penalty_top_k,
        )
        payload = {
            "metrics": metrics,
            "predictions": predictions,
            "grid_results": history,
        }
    else:
        result = evaluate_retrieval(
            module.model,
            relative,
            gallery,
            batch_size=cfg.train.val_batch_size,
            num_workers=cfg.data.num_workers,
            device=device,
            top_k=cfg.faiss.top_k,
            reference_penalty_weight=args.reference_penalty_weight,
            reference_penalty_top_k=args.reference_penalty_top_k,
        )
        payload = {"metrics": result.metrics, "predictions": result.predictions}

    print(json.dumps(payload["metrics"], indent=2))
    if "grid_results" in payload:
        print(json.dumps(payload["grid_results"], indent=2))
    if args.output:
        write_json(payload, args.output)


if __name__ == "__main__":
    main()
