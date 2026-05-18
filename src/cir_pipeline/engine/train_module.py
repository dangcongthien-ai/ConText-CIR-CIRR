"""PyTorch Lightning module for ConText-CIR-like training."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    pl = None

from cir_pipeline.config import AppConfig, config_to_dict
from cir_pipeline.engine.evaluator import evaluate_retrieval
from cir_pipeline.models.context_cir import ConTextCIR
from cir_pipeline.models.losses import retrieval_contrastive_loss, text_concept_consistency_loss
from cir_pipeline.text.token_alignment import align_noun_phrases, align_single_phrase


class CIRLightningModule(pl.LightningModule if pl else torch.nn.Module):
    """Lightning training module with retrieval and Text CC losses."""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = ConTextCIR(cfg.model)
        self.model.train()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters({"config": config_to_dict(cfg)})

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        return self.model.forward_query(images, texts)

    def _compute_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        reference = batch["reference_image"]
        target = batch["target_image"]
        captions = [str(text or "") for text in batch["caption"]]
        query_out = self.model.forward_with_attention(reference, captions)
        query_emb = query_out["embeddings"]
        target_emb = self.model.encode_gallery(target)
        query_neg = self.model.encode_gallery(reference) if self.cfg.loss.use_query_image_negative else None
        loss_cont = retrieval_contrastive_loss(
            query_emb,
            target_emb,
            query_neg,
            temperature=self.cfg.loss.temperature,
            use_query_image_negative=self.cfg.loss.use_query_image_negative,
        )
        loss_cc = self._compute_text_cc(batch, query_out)
        loss_total = loss_cont + self.cfg.loss.lambda_cc * loss_cc
        return {"loss_total": loss_total, "loss_cont": loss_cont, "loss_cc": loss_cc}

    def _compute_text_cc(self, batch: dict[str, Any], query_out: dict[str, Any]) -> torch.Tensor:
        if self.cfg.loss.lambda_cc <= 0:
            return query_out["embeddings"].new_zeros(())
        captions = [str(text or "") for text in batch["caption"]]
        noun_phrases = batch.get("noun_phrases") or [[] for _ in captions]
        full_token_indices: list[list[int]] = []
        iso_token_indices: list[list[int]] = []
        iso_texts: list[str] = []
        sample_indices: list[int] = []
        for sample_idx, (caption, phrases) in enumerate(zip(captions, noun_phrases)):
            phrases = list(phrases or [])[: self.cfg.loss.max_noun_phrases]
            aligned = align_noun_phrases(caption, phrases, self.model.tokenizer)
            for phrase, token_idx in zip(phrases, aligned):
                if not token_idx:
                    continue
                iso_idx = align_single_phrase(phrase, phrase, self.model.tokenizer)
                if not iso_idx:
                    continue
                full_token_indices.append(token_idx)
                iso_token_indices.append(iso_idx)
                iso_texts.append(phrase)
                sample_indices.append(sample_idx)

        if not iso_texts:
            return query_out["embeddings"].new_zeros(())
        iso_images = batch["reference_image"][torch.tensor(sample_indices, device=batch["reference_image"].device)]
        iso_out = self.model.forward_with_attention(iso_images, iso_texts)
        return text_concept_consistency_loss(
            query_out["attention_maps"],
            iso_out["attention_maps"],
            full_token_indices,
            iso_token_indices,
            sample_indices,
            epsilon=self.cfg.loss.epsilon_cc,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        losses = self._compute_losses(batch)
        batch_size = int(batch["reference_image"].shape[0])
        self.log(
            "train/loss_total",
            losses["loss_total"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("train/loss_cont", losses["loss_cont"], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_cc", losses["loss_cc"], on_step=True, on_epoch=True, batch_size=batch_size)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"] if self.trainer.optimizers else 0.0
        self.log("lr", lr, on_step=True, on_epoch=False, batch_size=batch_size)
        return losses["loss_total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        if batch["target_image"] is None:
            return None
        losses = self._compute_losses(batch)
        batch_size = int(batch["reference_image"].shape[0])
        self.log(
            "val/loss",
            losses["loss_total"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return losses

    def on_before_optimizer_step(self, optimizer) -> None:
        if not hasattr(self, "log"):
            return
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        self.log("train/grad_norm", norm, on_step=True, on_epoch=False, batch_size=1)

    def on_fit_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        train_dataset = getattr(datamodule, "train_dataset", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        metrics: dict[str, float] = {}
        if train_dataset is not None:
            metrics["num_samples_train"] = float(len(train_dataset))
        if val_dataset is not None:
            metrics["num_samples_val"] = float(len(val_dataset))
        self._log_direct_metrics(metrics)

    def _log_direct_metrics(self, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        logger = getattr(self, "logger", None)
        if logger and hasattr(logger, "log_metrics"):
            logger.log_metrics(metrics, step=0)

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False) or getattr(self.trainer, "fast_dev_run", False):
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        relative = datamodule.relative_dataset("val")
        gallery = datamodule.gallery_dataset("val")
        result = evaluate_retrieval(
            self.model,
            relative,
            gallery,
            batch_size=self.cfg.train.val_batch_size,
            num_workers=self.cfg.data.num_workers,
            device=self.device,
            top_k=self.cfg.faiss.top_k,
        )
        for key, value in result.metrics.items():
            metric_name = f"val/{key}" if not key.startswith("num_") else key
            self.log(
                metric_name,
                value,
                prog_bar=key in {"recall@10", "recall_subset@1"},
                batch_size=1,
            )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )

        total_steps = max(1, int(getattr(self.trainer, "estimated_stepping_batches", 1)))
        warmup_steps = int(total_steps * self.cfg.train.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
