"""Read-only CIRR dataset inspection utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cir_pipeline.config import AppConfig
from cir_pipeline.utils.io import normalize_rel_path, read_json


def _split_paths(cfg: AppConfig, split: str) -> tuple[Path, Path, Path]:
    metadata_root = Path(cfg.data.metadata_root)
    return (
        metadata_root / "image_splits" / f"split.rc2.{split}.json",
        metadata_root / "captions" / f"cap.rc2.{split}.json",
        metadata_root / "captions_ext" / f"cap.ext.rc2.{split}.json",
    )


def _iter_pngs(folder: Path):
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".png"):
                yield Path(root) / name


def inspect_cirr_files(cfg: AppConfig, scan_extra: bool = False) -> dict[str, Any]:
    """Inspect CIRR metadata and optional extra images without modifying data."""

    root = Path(cfg.data.root)
    summary: dict[str, Any] = {"root": str(root), "splits": {}}
    for split in ("train", "val", "test1"):
        split_path, captions_path, ext_path = _split_paths(cfg, split)
        split_map = read_json(split_path)
        captions = read_json(captions_path)
        ext_rows = read_json(ext_path) if ext_path.exists() else []
        missing = []
        for image_name, rel_path in split_map.items():
            path = root / normalize_rel_path(str(rel_path))
            if not path.exists():
                missing.append({"image_name": image_name, "path": str(path)})

        empty_captions = [
            row.get("pairid")
            for row in captions
            if not str(row.get("caption") or "").strip()
        ]
        unknown_refs = []
        unknown_targets = []
        for row in captions:
            ref = row.get("reference")
            target = row.get("target_hard")
            if ref and ref not in split_map:
                unknown_refs.append({"pairid": row.get("pairid"), "reference": ref})
            if target and target not in split_map:
                unknown_targets.append({"pairid": row.get("pairid"), "target": target})

        extra_count = None
        if scan_extra:
            declared = {str((root / normalize_rel_path(path)).resolve()) for path in split_map.values()}
            folder = {
                "train": Path(cfg.data.image_root_train),
                "val": Path(cfg.data.image_root_val),
                "test1": Path(cfg.data.image_root_test1),
            }[split]
            actual = {str(path.resolve()) for path in _iter_pngs(folder)} if folder.exists() else set()
            extra_count = len(actual - declared)

        summary["splits"][split] = {
            "official_images": len(split_map),
            "caption_pairs": len(captions),
            "extended_caption_rows": len(ext_rows),
            "missing_images": missing,
            "missing_image_count": len(missing),
            "empty_caption_pairids": empty_captions,
            "empty_caption_count": len(empty_captions),
            "unknown_references": unknown_refs,
            "unknown_reference_count": len(unknown_refs),
            "unknown_targets": unknown_targets,
            "unknown_target_count": len(unknown_targets),
            "extra_png_count": extra_count,
        }
    return summary
