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


PLACEHOLDER_RE = re.compile(r"^\s*\[cr\d+\]\s*", re.IGNORECASE)


def _is_placeholder_caption(text: str) -> bool:
    lowered = PLACEHOLDER_RE.sub("", text.strip()).lower()
    return lowered in {"", "nothing worth mentioning", "covered in query"}


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
        pieces: list[str] = []
        if isinstance(caption_extend, dict):
            values = caption_extend.values()
        elif isinstance(caption_extend, list):
            values = caption_extend
        else:
            values = []
        for value in values:
            text = str(value or "").strip()
            if not text or _is_placeholder_caption(text):
                continue
            pieces.append(PLACEHOLDER_RE.sub("", text).strip())
        return " ".join(pieces) if pieces else caption

    def _load_relative_samples(self) -> list[dict[str, Any]]:
        rows = self._load_captions()
        samples: list[dict[str, Any]] = []
        empty_count = 0
        skipped_count = 0

        for row in rows:
            caption = self._merged_caption(row)
            if not caption:
                empty_count += 1
                if self.split == "train" and self.data_cfg.skip_empty_captions:
                    skipped_count += 1
                    continue

            reference = str(row.get("reference"))
            target = row.get("target_hard")
            if reference not in self.image_splits:
                warnings.warn(f"Skipping pair {row.get('pairid')}: unknown reference {reference}")
                continue
            if target is not None and str(target) not in self.image_splits:
                warnings.warn(f"Skipping pair {row.get('pairid')}: unknown target {target}")
                continue

            img_set = row.get("img_set") or {}
            members = img_set.get("members", []) if isinstance(img_set, dict) else []
            sample = {
                "pair_id": row.get("pairid"),
                "reference_name": reference,
                "target_name": str(target) if target is not None else None,
                "target_soft": row.get("target_soft") or {},
                "caption": caption,
                "img_set": list(members),
                "img_set_id": img_set.get("id") if isinstance(img_set, dict) else None,
            }
            samples.append(sample)

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

        caption = sample["caption"]
        noun_phrases = (
            self.noun_phrase_parser.extract(caption)
            if self.noun_phrase_parser is not None and caption
            else []
        )
        sample.update(
            {
                "reference_image": self._load_image(sample["reference_name"]),
                "reference_path": str(self.resolve_image_path(sample["reference_name"])),
                "target_image": self._load_image(sample["target_name"]) if sample["target_name"] else None,
                "target_path": str(self.resolve_image_path(sample["target_name"]))
                if sample["target_name"]
                else None,
                "noun_phrases": noun_phrases,
                "np_spans": None,
                "split": self.split,
            }
        )
        return sample
