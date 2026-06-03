"""PyTorch Lightning DataModule for CIRR."""

from __future__ import annotations

from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    pl = None

from cir_pipeline.config import AppConfig
from cir_pipeline.data.cirr_dataset import CIRRDataset, SplitName
from cir_pipeline.data.collate import cirr_collate
from cir_pipeline.data.transforms import build_clip_transform
from cir_pipeline.text.noun_phrase_parser import NounPhraseParser


class CIRRDataModule(pl.LightningDataModule if pl else object):
    """Create CIRR relative-query and gallery dataloaders."""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.noun_parser = (
            None
            if cfg.data.noun_phrase_backend == "none" or cfg.loss.lambda_cc <= 0
            else NounPhraseParser(
                max_phrases=cfg.loss.max_noun_phrases,
                backend=cfg.data.noun_phrase_backend,
                stanza_resources_dir=cfg.data.stanza_resources_dir,
                stanza_download_method=cfg.data.stanza_download_method,
                stanza_np_extractor=getattr(cfg.data, "noun_phrase_extractor", "constituency"),
            )
        )
        self.train_dataset: CIRRDataset | None = None
        self.val_dataset: CIRRDataset | None = None
        self.test_dataset: CIRRDataset | None = None

    def _build_transform(self, train: bool) -> object:
        return build_clip_transform(
            train=train,
            backbone=self.cfg.model.backbone,
            transform_name=getattr(self.cfg.data, "image_transform", "clip"),
            target_pad_ratio=float(getattr(self.cfg.data, "target_pad_ratio", 1.25)),
        )

    def setup(self, stage: str | None = None) -> None:
        train_tf = self._build_transform(train=True)
        eval_tf = self._build_transform(train=False)
        if stage in {None, "fit"}:
            self.train_dataset = CIRRDataset(
                self.cfg.data, "train", "relative", train_tf, self.noun_parser
            )
            self.val_dataset = CIRRDataset(
                self.cfg.data, "val", "relative", eval_tf, None
            )
        if stage in {None, "validate"} and self.val_dataset is None:
            self.val_dataset = CIRRDataset(
                self.cfg.data, "val", "relative", eval_tf, None
            )
        if stage in {None, "test", "predict"}:
            self.test_dataset = CIRRDataset(
                self.cfg.data, "test1", "relative", eval_tf, None
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=self.cfg.data.num_workers > 0,
            pin_memory=True,
            collate_fn=cirr_collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('validate') before val_dataloader().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=self.cfg.data.num_workers > 0,
            pin_memory=True,
            collate_fn=cirr_collate,
        )

    def relative_dataset(self, split: SplitName) -> CIRRDataset:
        return CIRRDataset(
            self.cfg.data,
            split,
            "relative",
            self._build_transform(train=False),
            None,
        )

    def gallery_dataset(self, split: SplitName) -> CIRRDataset:
        return CIRRDataset(
            self.cfg.data,
            split,
            "gallery",
            self._build_transform(train=False),
            None,
        )
