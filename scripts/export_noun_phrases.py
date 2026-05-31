from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from cir_pipeline.config import AppConfig, load_config
from cir_pipeline.text.noun_phrase_parser import NounPhraseParser, prepare_stanza_resources
from cir_pipeline.utils.io import ensure_dir, read_json, write_json

SPLITS: tuple[str, ...] = ("train", "val", "test1")
PLACEHOLDER_RE = re.compile(r"^\s*\[cr\d+\]\s*", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export noun phrases for CIRR captions with Stanza and/or regex.")
    parser.add_argument("--config", default="configs/cirr_context_cir_cl_cc.yaml")
    parser.add_argument("--split", default="all", choices=[*SPLITS, "all"])
    parser.add_argument("--backend", default="both", choices=["stanza", "regex", "both"])
    parser.add_argument("--output_dir", default="outputs/noun_phrases")
    parser.add_argument("--max_phrases", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Allow Stanza output to fall back to regex if Stanza is unavailable.",
    )
    parser.add_argument(
        "--prepare_stanza",
        action="store_true",
        help="Download/prepare Stanza resources before exporting.",
    )
    parser.add_argument(
        "--per_backend_dirs",
        action="store_true",
        help="Also write outputs/stanza/stanza_output_<split>.json and outputs/regex/regex_output_<split>.json.",
    )
    return parser.parse_args()


def _selected_splits(split: str) -> list[str]:
    return list(SPLITS) if split == "all" else [split]


def _selected_backends(backend: str) -> list[str]:
    return ["stanza", "regex"] if backend == "both" else [backend]


def _is_placeholder_caption(text: str) -> bool:
    lowered = PLACEHOLDER_RE.sub("", text.strip()).lower()
    return lowered in {"", "nothing worth mentioning", "covered in query"}


def _caption_extend_values(caption_extend: Any) -> list[Any]:
    if isinstance(caption_extend, dict):
        return list(caption_extend.values())
    if isinstance(caption_extend, list):
        return caption_extend
    return []


def _clean_extended_caption(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or _is_placeholder_caption(text):
        return None
    return PLACEHOLDER_RE.sub("", text).strip()


def _load_split_samples(cfg: AppConfig, split: str) -> list[dict[str, Any]]:
    metadata_root = Path(cfg.data.metadata_root)
    image_splits, caption_rows = _load_split_metadata(metadata_root, split)
    ext_by_pairid = _load_extended_captions(metadata_root, split, cfg.data.use_extended_captions)

    samples: list[dict[str, Any]] = []
    for row in caption_rows:
        sample = _sample_from_caption_row(row, image_splits, ext_by_pairid, cfg, split)
        if sample is not None:
            samples.append(sample)
    return samples


def _load_split_metadata(metadata_root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_splits_path = metadata_root / "image_splits" / f"split.rc2.{split}.json"
    captions_path = metadata_root / "captions" / f"cap.rc2.{split}.json"
    image_splits = read_json(image_splits_path)
    caption_rows = read_json(captions_path)
    if not isinstance(image_splits, dict):
        raise ValueError(f"Expected image split manifest to be a dict: {image_splits_path}")
    if not isinstance(caption_rows, list):
        raise ValueError(f"Expected captions file to be a list: {captions_path}")
    return image_splits, caption_rows


def _load_extended_captions(
    metadata_root: Path,
    split: str,
    use_extended: bool,
) -> dict[str, dict[str, Any]]:
    if not use_extended:
        return {}
    ext_path = metadata_root / "captions_ext" / f"cap.ext.rc2.{split}.json"
    if not ext_path.exists():
        return {}
    ext_rows = read_json(ext_path)
    return {str(row.get("pairid")): row for row in ext_rows}


def _sample_from_caption_row(
    row: dict[str, Any],
    image_splits: dict[str, Any],
    ext_by_pairid: dict[str, dict[str, Any]],
    cfg: AppConfig,
    split: str,
) -> dict[str, Any] | None:
    caption = _merged_caption(row, ext_by_pairid, cfg.data.use_extended_captions)
    if not caption and split == "train" and cfg.data.skip_empty_captions:
        return None

    reference = str(row.get("reference"))
    target = row.get("target_hard")
    if not _has_valid_images(reference, target, image_splits):
        return None

    img_set = row.get("img_set") or {}
    members = img_set.get("members", []) if isinstance(img_set, dict) else []
    return {
        "pair_id": row.get("pairid"),
        "reference_name": reference,
        "target_name": str(target) if target is not None else None,
        "caption": caption,
        "target_soft": row.get("target_soft") or {},
        "img_set_id": img_set.get("id") if isinstance(img_set, dict) else None,
        "img_set": list(members),
    }


def _has_valid_images(reference: str, target: Any, image_splits: dict[str, Any]) -> bool:
    if reference not in image_splits:
        return False
    return target is None or str(target) in image_splits


def _merged_caption(row: dict[str, Any], ext_by_pairid: dict[str, dict[str, Any]], use_extended: bool) -> str:
    caption = str(row.get("caption") or "").strip()
    if not use_extended:
        return caption
    ext = ext_by_pairid.get(str(row.get("pairid")), {})
    caption_extend = ext.get("caption_extend") or {}
    pieces = [
        cleaned
        for value in _caption_extend_values(caption_extend)
        if (cleaned := _clean_extended_caption(value))
    ]
    return " ".join(pieces) if pieces else caption


def _build_parser(cfg: AppConfig, backend: str, max_phrases: int) -> NounPhraseParser:
    return NounPhraseParser(
        max_phrases=max_phrases,
        backend=backend,
        stanza_resources_dir=cfg.data.stanza_resources_dir,
        stanza_download_method=cfg.data.stanza_download_method,
        stanza_np_extractor=getattr(cfg.data, "noun_phrase_extractor", "constituency"),
    )


def _preflight_stanza(parser: NounPhraseParser, allow_fallback: bool) -> None:
    if allow_fallback:
        return
    if parser._ensure_stanza() is None:
        raise RuntimeError(
            "Stanza backend is unavailable. Prepare resources first or rerun with --allow_fallback. "
            "For this repo, try: python scripts/export_noun_phrases.py --backend stanza --prepare_stanza"
        )


def _extract(parser: NounPhraseParser, backend: str, caption: str, allow_fallback: bool) -> list[str]:
    if backend == "stanza" and not allow_fallback:
        return parser._extract_stanza(caption)[: parser.max_phrases]
    return parser.extract(caption)[: parser.max_phrases]


def _sample_row(
    sample: dict[str, Any],
    phrases: list[str],
    backend: str,
) -> dict[str, Any]:
    return {
        "pair_id": sample.get("pair_id"),
        "reference_name": sample.get("reference_name"),
        "target_name": sample.get("target_name"),
        "caption": sample.get("caption"),
        "noun_phrases": phrases,
        "num_noun_phrases": len(phrases),
        "img_set_id": sample.get("img_set_id"),
        "img_set": sample.get("img_set", []),
        "backend": backend,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(row["num_noun_phrases"]) for row in rows]
    if not counts:
        return {
            "num_samples": 0,
            "num_empty": 0,
            "avg_phrases": 0.0,
            "max_phrases": 0,
        }
    return {
        "num_samples": len(rows),
        "num_empty": sum(1 for count in counts if count == 0),
        "avg_phrases": sum(counts) / len(counts),
        "max_phrases": max(counts),
    }


def _export_backend_split(
    cfg: AppConfig,
    split: str,
    backend: str,
    output_dir: Path,
    max_phrases: int,
    limit: int | None,
    allow_fallback: bool,
    per_backend_dirs: bool,
) -> Path:
    parser = _build_parser(cfg, backend, max_phrases)
    if backend == "stanza":
        _preflight_stanza(parser, allow_fallback)

    all_samples = _load_split_samples(cfg, split)
    samples = all_samples[:limit] if limit is not None else all_samples
    rows: list[dict[str, Any]] = []
    progress = tqdm(samples, desc=f"{backend} {split}", unit="caption")
    for sample in progress:
        caption = str(sample.get("caption") or "")
        phrases = _extract(parser, backend, caption, allow_fallback)
        rows.append(_sample_row(sample, phrases, backend))

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg.data.dataset,
        "split": split,
        "backend": backend,
        "allow_fallback": allow_fallback,
        "max_phrases": max_phrases,
        "summary": _summary(rows),
        "samples": rows,
    }
    output_path = output_dir / f"cirr_{split}_{backend}.json"
    write_json(payload, output_path)
    if per_backend_dirs:
        legacy_dir = ensure_dir(Path(cfg.project.output_dir) / backend)
        write_json(payload, legacy_dir / f"{backend}_output_{split}.json")
    print(f"{backend} {split}: {output_path}")
    return output_path


def _write_comparison(
    cfg: AppConfig,
    split: str,
    output_dir: Path,
    backend_payloads: dict[str, Any],
) -> Path | None:
    if set(backend_payloads) != {"stanza", "regex"}:
        return None
    stanza_rows = {str(row["pair_id"]): row for row in backend_payloads["stanza"]["samples"]}
    regex_rows = {str(row["pair_id"]): row for row in backend_payloads["regex"]["samples"]}
    compared = []
    for pair_id, stanza_row in stanza_rows.items():
        regex_row = regex_rows.get(pair_id)
        if regex_row is None:
            continue
        stanza_phrases = stanza_row["noun_phrases"]
        regex_phrases = regex_row["noun_phrases"]
        compared.append(
            {
                "pair_id": stanza_row["pair_id"],
                "reference_name": stanza_row["reference_name"],
                "target_name": stanza_row["target_name"],
                "caption": stanza_row["caption"],
                "stanza_phrases": stanza_phrases,
                "regex_phrases": regex_phrases,
                "same_phrases": stanza_phrases == regex_phrases,
            }
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg.data.dataset,
        "split": split,
        "backends": ["stanza", "regex"],
        "summary": {
            "num_samples": len(compared),
            "num_same": sum(1 for row in compared if row["same_phrases"]),
        },
        "samples": compared,
    }
    output_path = output_dir / f"cirr_{split}_stanza_regex.json"
    write_json(payload, output_path)
    print(f"compare {split}: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    max_phrases = int(args.max_phrases or max(1, cfg.loss.max_noun_phrases))
    output_dir = ensure_dir(args.output_dir)
    backends = _selected_backends(args.backend)
    if args.prepare_stanza and "stanza" in backends:
        prepare_stanza_resources(cfg.data.stanza_resources_dir)

    for split in _selected_splits(args.split):
        backend_payloads: dict[str, Any] = {}
        for backend in backends:
            output_path = _export_backend_split(
                cfg=cfg,
                split=split,
                backend=backend,
                output_dir=output_dir,
                max_phrases=max_phrases,
                limit=args.limit,
                allow_fallback=args.allow_fallback,
                per_backend_dirs=args.per_backend_dirs,
            )
            with output_path.open("r", encoding="utf-8") as handle:
                backend_payloads[backend] = json.load(handle)
        _write_comparison(cfg, split, output_dir, backend_payloads)


if __name__ == "__main__":
    main()
