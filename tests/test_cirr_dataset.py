from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("torch")

from cir_pipeline.config import AppConfig, DataConfig
from cir_pipeline.data.cirr_dataset import CIRRDataset
from cir_pipeline.text.noun_phrase_parser import CaptionParseResult, PhraseRecord


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path)


def _tiny_cfg(tmp_path: Path) -> AppConfig:
    root = tmp_path / "CIRR"
    for split_dir in ("train/0", "dev", "test1"):
        (root / split_dir).mkdir(parents=True, exist_ok=True)
    _write_image(root / "train/0/ref.png")
    _write_image(root / "train/0/tgt.png")
    _write_image(root / "train/0/extra.png")
    _write_image(root / "dev/refv.png")
    _write_image(root / "dev/tgtv.png")
    _write_image(root / "test1/reft.png")

    metadata = root / "cirr"
    _write_json(
        metadata / "image_splits/split.rc2.train.json",
        {"ref": "./train/0/ref.png", "tgt": "./train/0/tgt.png"},
    )
    _write_json(
        metadata / "image_splits/split.rc2.val.json",
        {"refv": "./dev/refv.png", "tgtv": "./dev/tgtv.png"},
    )
    _write_json(
        metadata / "image_splits/split.rc2.test1.json",
        {"reft": "./test1/reft.png"},
    )
    _write_json(
        metadata / "captions/cap.rc2.train.json",
        [
            {
                "pairid": 1,
                "reference": "ref",
                "target_hard": "tgt",
                "target_soft": {"tgt": 1.0},
                "caption": "make the red square darker",
                "img_set": {"members": ["ref", "tgt"]},
            },
            {
                "pairid": 2,
                "reference": "ref",
                "target_hard": "tgt",
                "caption": "",
                "img_set": {"members": ["ref", "tgt"]},
            },
        ],
    )
    _write_json(
        metadata / "captions/cap.rc2.val.json",
        [
            {
                "pairid": 3,
                "reference": "refv",
                "target_hard": "tgtv",
                "target_soft": {"tgtv": 1.0},
                "caption": "show the target",
                "img_set": {"members": ["refv", "tgtv"]},
            }
        ],
    )
    _write_json(
        metadata / "captions/cap.rc2.test1.json",
        [{"pairid": 4, "reference": "reft", "caption": "", "img_set": {"members": ["reft"]}}],
    )
    for split in ("train", "val", "test1"):
        _write_json(metadata / f"captions_ext/cap.ext.rc2.{split}.json", [])

    cfg = AppConfig()
    cfg.data = DataConfig(
        root=str(root),
        image_root_train=str(root / "train"),
        image_root_val=str(root / "dev"),
        image_root_test1=str(root / "test1"),
        metadata_root=str(metadata),
        skip_empty_captions=True,
        num_workers=0,
    )
    return cfg


def test_dataset_uses_official_split_without_rglob(tmp_path, monkeypatch):
    cfg = _tiny_cfg(tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("Path.rglob must not be used by CIRRDataset")

    monkeypatch.setattr(Path, "rglob", explode)
    dataset = CIRRDataset(cfg.data, "train", "relative", transform=None)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["reference_name"] == "ref"
    assert sample["target_name"] == "tgt"
    assert dataset.resolve_image_path("ref").name == "ref.png"


def test_test1_empty_caption_does_not_crash(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    dataset = CIRRDataset(cfg.data, "test1", "relative", transform=None)
    sample = dataset[0]
    assert sample["caption"] == ""
    assert sample["target_image"] is None


def test_dataset_includes_parser_metadata(tmp_path):
    cfg = _tiny_cfg(tmp_path)

    class StubParser:
        def analyze(self, caption: str) -> CaptionParseResult:
            return CaptionParseResult(
                noun_phrases=["red square"],
                phrase_records=[PhraseRecord(text="red square", confidence=0.95, extractor="test")],
                edit_actions=[
                    {
                        "action": "make",
                        "object": "the red square",
                        "attribute": "darker",
                        "target": None,
                        "location": None,
                        "raw_clause": caption,
                        "confidence": 0.9,
                    }
                ],
                parser_warnings=["used_test_parser"],
                phrase_extractor="test",
                normalized_caption=caption,
            )

    dataset = CIRRDataset(cfg.data, "train", "relative", transform=None, noun_phrase_parser=StubParser())
    sample = dataset[0]
    assert sample["noun_phrases"] == ["red square"]
    assert sample["phrase_records"][0]["text"] == "red square"
    assert sample["edit_actions"][0]["action"] == "make"
    assert sample["parser_warnings"] == ["used_test_parser"]
    assert sample["phrase_extractor"] == "test"


def test_dataset_prefers_exported_noun_phrase_cache(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    cache_dir = tmp_path / "noun_phrases"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.noun_phrase_backend = "stanza"
    cfg.data.noun_phrase_extractor = "constituency_minimal_np"
    cfg.data.noun_phrase_cache_dir = str(cache_dir)
    _write_json(
        cache_dir / "cirr_train_stanza.json",
        {
            "dataset": "cirr",
            "split": "train",
            "backend": "stanza",
            "noun_phrase_extractor": "constituency_minimal_np",
            "samples": [
                {
                    "pair_id": 1,
                    "noun_phrases": ["cached red square"],
                    "phrase_records": [
                        {
                            "text": "cached red square",
                            "confidence": 0.95,
                            "extractor": "constituency_minimal_np",
                        }
                    ],
                    "parser_warnings": ["loaded_from_cache"],
                    "phrase_extractor": "constituency_minimal_np",
                    "edit_actions": [{"action": "make", "object": "square"}],
                }
            ],
        },
    )

    class ExplodingParser:
        def analyze(self, caption: str) -> CaptionParseResult:
            raise AssertionError(f"Parser should not run when cache exists: {caption}")

    dataset = CIRRDataset(cfg.data, "train", "relative", transform=None, noun_phrase_parser=ExplodingParser())
    sample = dataset[0]
    assert sample["noun_phrases"] == ["cached red square"]
    assert sample["phrase_records"][0]["text"] == "cached red square"
    assert sample["parser_warnings"] == ["loaded_from_cache"]
    assert sample["phrase_extractor"] == "constituency_minimal_np"
    assert sample["edit_actions"][0]["action"] == "make"
