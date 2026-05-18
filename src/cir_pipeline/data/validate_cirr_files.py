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


def _missing_images(root: Path, split_map: dict[str, str]) -> list[dict[str, str]]:
    missing = []
    for image_name, rel_path in split_map.items():
        path = root / normalize_rel_path(str(rel_path))
        if not path.exists():
            missing.append({"image_name": image_name, "path": str(path)})
    return missing


def _empty_caption_pairids(captions: list[dict[str, Any]]) -> list[Any]:
    return [
        row.get("pairid")
        for row in captions
        if not str(row.get("caption") or "").strip()
    ]


def _unknown_caption_names(
    captions: list[dict[str, Any]],
    split_map: dict[str, str],
    field: str,
    output_key: str,
) -> list[dict[str, Any]]:
    unknown = []
    for row in captions:
        value = row.get(field)
        if value and value not in split_map:
            unknown.append({"pairid": row.get("pairid"), output_key: value})
    return unknown


def _extra_png_count(cfg: AppConfig, split: str, root: Path, split_map: dict[str, str]) -> int | None:
    folder = {
        "train": Path(cfg.data.image_root_train),
        "val": Path(cfg.data.image_root_val),
        "test1": Path(cfg.data.image_root_test1),
    }[split]
    if not folder.exists():
        return None
    declared = {str((root / normalize_rel_path(path)).resolve()) for path in split_map.values()}
    actual = {str(path.resolve()) for path in _iter_pngs(folder)}
    return len(actual - declared)


def _inspect_split(cfg: AppConfig, split: str, scan_extra: bool) -> dict[str, Any]:
    root = Path(cfg.data.root)
    split_path, captions_path, ext_path = _split_paths(cfg, split)
    split_map = read_json(split_path)
    captions = read_json(captions_path)
    ext_rows = read_json(ext_path) if ext_path.exists() else []
    missing = _missing_images(root, split_map)
    empty_captions = _empty_caption_pairids(captions)
    unknown_refs = _unknown_caption_names(captions, split_map, "reference", "reference")
    unknown_targets = _unknown_caption_names(captions, split_map, "target_hard", "target")
    extra_count = _extra_png_count(cfg, split, root, split_map) if scan_extra else None

    return {
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


def inspect_cirr_files(cfg: AppConfig, scan_extra: bool = False) -> dict[str, Any]:
    """Inspect CIRR metadata and optional extra images without modifying data."""

    root = Path(cfg.data.root)
    summary: dict[str, Any] = {"root": str(root), "splits": {}}
    for split in ("train", "val", "test1"):
        summary["splits"][split] = _inspect_split(cfg, split, scan_extra)
    return summary
