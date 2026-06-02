"""CIRR dataset readers backed only by official metadata manifests."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from PIL import Image
from torch.utils.data import Dataset

from cir_pipeline.config import DataConfig
from cir_pipeline.text.noun_phrase_parser import NounPhraseParser
from cir_pipeline.utils.io import normalize_rel_path, read_json

SplitName = Literal["train", "val", "test1"]
ModeName = Literal["relative", "gallery"]

# CIRR contains a few legitimate high-resolution images. The loader resizes them
# immediately through the transform, so PIL's decompression-bomb guard is noisy
# for this trusted local dataset.
Image.MAX_IMAGE_PIXELS = None

PLACEHOLDER_RE = re.compile(r"^\s*\[cr\d+\]\s*", re.IGNORECASE)


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


class CIRRDataset(Dataset):
    """CIRR samples resolved from image_splits plus captions, never folder scans."""

    def __init__(
        self,
        data_cfg: DataConfig,
        split: SplitName,
        mode: ModeName = "relative",
        transform: Optional[Callable[[Image.Image], Any]] = None,
        noun_phrase_parser: Optional[NounPhraseParser] = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.split = split
        self.mode = mode
        self.transform = transform
        self.noun_phrase_parser = noun_phrase_parser
        self.root = Path(data_cfg.root)
        self.metadata_root = Path(data_cfg.metadata_root)
        self.noun_phrase_cache = self._load_noun_phrase_cache()

        self.image_splits: dict[str, str] = self._load_image_split()
        self.image_names = list(self.image_splits.keys())
        self.ext_by_pairid = self._load_extended_captions() if data_cfg.use_extended_captions else {}

        if mode == "gallery":
            self.samples: list[dict[str, Any]] = [
                {"image_name": name, "image_path": str(self.resolve_image_path(name))}
                for name in self.image_names
            ]
        elif mode == "relative":
            self.samples = self._load_relative_samples()
        else:
            raise ValueError(f"Unsupported CIRR mode: {mode}")

    def _phrase_cache_candidates(self) -> list[Path]:
        backend = str(getattr(self.data_cfg, "noun_phrase_backend", "none") or "none")
        if backend == "none":
            return []
        base_name = f"cirr_{self.split}_{backend}.json"
        candidates: list[Path] = []
        cache_dir = getattr(self.data_cfg, "noun_phrase_cache_dir", None)
        if cache_dir:
            candidates.append(Path(cache_dir) / base_name)
        legacy_dir = Path("outputs") / "noun_phrases"
        legacy_path = legacy_dir / base_name
        if legacy_path not in candidates:
            candidates.append(legacy_path)
        return candidates

    def _payload_phrase_extractor(self, payload: dict[str, Any]) -> str | None:
        extractor = payload.get("noun_phrase_extractor")
        if extractor:
            return str(extractor)
        samples = payload.get("samples")
        if isinstance(samples, list) and samples:
            sample_extractor = samples[0].get("phrase_extractor")
            if sample_extractor:
                return str(sample_extractor)
        return None

    def _load_noun_phrase_cache(self) -> dict[str, dict[str, Any]]:
        if self.mode != "relative" or not bool(getattr(self.data_cfg, "use_exported_noun_phrases", True)):
            return {}
        for path in self._phrase_cache_candidates():
            if not path.exists():
                continue
            payload = read_json(path)
            if not isinstance(payload, dict):
                warnings.warn(f"Ignoring noun phrase cache with unexpected format: {path}")
                continue
            backend = str(payload.get("backend") or "")
            expected_backend = str(getattr(self.data_cfg, "noun_phrase_backend", "none") or "none")
            if backend and expected_backend not in {"auto", backend}:
                warnings.warn(
                    f"Ignoring noun phrase cache {path}: backend {backend!r} does not match config {expected_backend!r}."
                )
                continue
            payload_extractor = self._payload_phrase_extractor(payload)
            expected_extractor = str(getattr(self.data_cfg, "noun_phrase_extractor", "constituency") or "constituency")
            if payload_extractor and payload_extractor != expected_extractor:
                warnings.warn(
                    f"Ignoring noun phrase cache {path}: extractor {payload_extractor!r} does not match "
                    f"config {expected_extractor!r}."
                )
                continue
            rows = payload.get("samples")
            if not isinstance(rows, list):
                warnings.warn(f"Ignoring noun phrase cache without 'samples' list: {path}")
                continue
            if not any(isinstance(row, dict) and "phrase_records" in row for row in rows):
                warnings.warn(
                    f"Ignoring legacy noun phrase cache without phrase_records: {path}. "
                    "Re-export noun phrases before training CL+CC."
                )
                continue
            cache: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                pair_id = row.get("pair_id")
                if pair_id is None:
                    continue
                cache[str(pair_id)] = row
            if cache:
                return cache
        return {}

    def _load_image_split(self) -> dict[str, str]:
        path = self.metadata_root / "image_splits" / f"split.rc2.{self.split}.json"
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Expected image split manifest to be a dict: {path}")
        return {str(name): str(rel_path) for name, rel_path in data.items()}

    def _load_captions(self) -> list[dict[str, Any]]:
        path = self.metadata_root / "captions" / f"cap.rc2.{self.split}.json"
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected captions file to be a list: {path}")
        return data

    def _load_extended_captions(self) -> dict[str, dict[str, Any]]:
        path = self.metadata_root / "captions_ext" / f"cap.ext.rc2.{self.split}.json"
        if not path.exists():
            warnings.warn(f"Extended captions requested but file is missing: {path}")
            return {}
        rows = read_json(path)
        return {str(row.get("pairid")): row for row in rows}

    def _merged_caption(self, row: dict[str, Any]) -> str:
        caption = str(row.get("caption") or "").strip()
        if not self.data_cfg.use_extended_captions:
            return caption

        ext = self.ext_by_pairid.get(str(row.get("pairid")), {})
        caption_extend = ext.get("caption_extend") or {}
        pieces = [
            cleaned
            for value in _caption_extend_values(caption_extend)
            if (cleaned := _clean_extended_caption(value))
        ]
        return " ".join(pieces) if pieces else caption

    def _should_skip_empty_caption(self, caption: str) -> bool:
        return not caption and self.split == "train" and self.data_cfg.skip_empty_captions

    def _names_are_valid(self, row: dict[str, Any], reference: str, target: Any) -> bool:
        if reference not in self.image_splits:
            warnings.warn(f"Skipping pair {row.get('pairid')}: unknown reference {reference}")
            return False
        if target is not None and str(target) not in self.image_splits:
            warnings.warn(f"Skipping pair {row.get('pairid')}: unknown target {target}")
            return False
        return True

    def _build_relative_sample(
        self,
        row: dict[str, Any],
        caption: str,
        reference: str,
        target: Any,
    ) -> dict[str, Any]:
        img_set = row.get("img_set") or {}
        members = img_set.get("members", []) if isinstance(img_set, dict) else []
        return {
            "pair_id": row.get("pairid"),
            "reference_name": reference,
            "target_name": str(target) if target is not None else None,
            "target_soft": row.get("target_soft") or {},
            "caption": caption,
            "img_set": list(members),
            "img_set_id": img_set.get("id") if isinstance(img_set, dict) else None,
        }

    def _load_relative_samples(self) -> list[dict[str, Any]]:
        rows = self._load_captions()
        samples: list[dict[str, Any]] = []
        empty_count = 0
        skipped_count = 0

        for row in rows:
            caption = self._merged_caption(row)
            if not caption:
                empty_count += 1
                if self._should_skip_empty_caption(caption):
                    skipped_count += 1
                    continue

            reference = str(row.get("reference"))
            target = row.get("target_hard")
            if not self._names_are_valid(row, reference, target):
                continue
            samples.append(self._build_relative_sample(row, caption, reference, target))

        if empty_count:
            warnings.warn(
                f"CIRR {self.split} has {empty_count} empty captions; "
                f"skipped {skipped_count} according to config."
            )
        return samples

    def resolve_image_path(self, image_name: str) -> Path:
        """Resolve an image name using only split.rc2.*.json."""

        if image_name not in self.image_splits:
            raise KeyError(f"Image {image_name!r} is not in split.rc2.{self.split}.json")
        return self.root / normalize_rel_path(self.image_splits[image_name])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_name: str):
        path = self.resolve_image_path(image_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing CIRR image for {image_name}: {path}")
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else image

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.samples[index])
        if self.mode == "gallery":
            image_name = sample["image_name"]
            sample.update(
                {
                    "image": self._load_image(image_name),
                    "image_path": str(self.resolve_image_path(image_name)),
                    "split": self.split,
                }
            )
            return sample

        cache_row = self.noun_phrase_cache.get(str(sample.get("pair_id")))
        parse_result = None
        noun_phrases: list[str]
        phrase_records: list[dict[str, Any]]
        parser_warnings: list[str]
        phrase_extractor: str | None
        edit_actions: list[dict[str, Any]]
        if cache_row is not None:
            noun_phrases = list(cache_row.get("noun_phrases") or [])
            phrase_records = list(cache_row.get("phrase_records") or [])
            parser_warnings = list(cache_row.get("parser_warnings") or [])
            phrase_extractor = cache_row.get("phrase_extractor")
            edit_actions = list(cache_row.get("edit_actions") or [])
        else:
            caption = sample["caption"]
            parse_result = (
                self.noun_phrase_parser.analyze(caption)
                if self.noun_phrase_parser is not None and caption
                else None
            )
            noun_phrases = parse_result.noun_phrases if parse_result is not None else []
            phrase_records = [record.to_dict() for record in parse_result.phrase_records] if parse_result is not None else []
            parser_warnings = list(parse_result.parser_warnings) if parse_result is not None else []
            phrase_extractor = parse_result.phrase_extractor if parse_result is not None else None
            edit_actions = list(parse_result.edit_actions) if parse_result is not None else []
        sample.update(
            {
                "reference_image": self._load_image(sample["reference_name"]),
                "reference_path": str(self.resolve_image_path(sample["reference_name"])),
                "target_image": self._load_image(sample["target_name"]) if sample["target_name"] else None,
                "target_path": str(self.resolve_image_path(sample["target_name"]))
                if sample["target_name"]
                else None,
                "noun_phrases": noun_phrases,
                "phrase_records": phrase_records,
                "parser_warnings": parser_warnings,
                "phrase_extractor": phrase_extractor,
                "edit_actions": edit_actions,
                "np_spans": None,
                "split": self.split,
            }
        )
        return sample
