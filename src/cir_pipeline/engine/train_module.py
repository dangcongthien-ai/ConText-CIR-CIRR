"""PyTorch Lightning module for ConText-CIR-like training."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class TextCCInputs:
    full_token_indices: list[list[int]]
    iso_token_indices: list[list[int]]
    iso_texts: list[str]
    sample_indices: list[int]


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

    def _compute_losses(
        self,
        batch: dict[str, Any],
        include_query_negative: bool = True,
        include_text_cc: bool = True,
    ) -> dict[str, torch.Tensor]:
        reference = batch["reference_image"]
        target = batch["target_image"]
        captions = [str(text or "") for text in batch["caption"]]
        query_out = self.model.forward_with_attention(reference, captions)
        query_emb = query_out["embeddings"]
        target_emb = self.model.encode_gallery(target)
        use_query_negative = self.cfg.loss.use_query_image_negative and include_query_negative
        query_neg = self.model.encode_gallery(reference) if use_query_negative else None
        loss_cont = retrieval_contrastive_loss(
            query_emb,
            target_emb,
            query_neg,
            temperature=self.cfg.loss.temperature,
            use_query_image_negative=use_query_negative,
            query_negative_weight=self.cfg.loss.query_negative_weight if include_query_negative else 0.0,
        )
        loss_cc = self._compute_text_cc(batch, query_out) if include_text_cc else query_emb.new_zeros(())
        loss_total = loss_cont + self.cfg.loss.lambda_cc * loss_cc
        return {"loss_total": loss_total, "loss_cont": loss_cont, "loss_cc": loss_cc}

    def _compute_text_cc(self, batch: dict[str, Any], query_out: dict[str, Any]) -> torch.Tensor:
        if self.cfg.loss.lambda_cc <= 0:
            return query_out["embeddings"].new_zeros(())
        captions = [str(text or "") for text in batch["caption"]]
        noun_phrases = batch.get("noun_phrases") or [[] for _ in captions]
        text_cc_inputs = self._build_text_cc_inputs(captions, noun_phrases)
        if not text_cc_inputs.iso_texts:
            return query_out["embeddings"].new_zeros(())

        iso_images = batch["reference_image"][
            torch.tensor(text_cc_inputs.sample_indices, device=batch["reference_image"].device)
        ]
        iso_out = self.model.forward_with_attention(iso_images, text_cc_inputs.iso_texts)
        return text_concept_consistency_loss(
            query_out["attention_maps"],
            iso_out["attention_maps"],
            text_cc_inputs.full_token_indices,
            text_cc_inputs.iso_token_indices,
            text_cc_inputs.sample_indices,
            epsilon=self.cfg.loss.epsilon_cc,
        )

    def _build_text_cc_inputs(
        self,
        captions: list[str],
        noun_phrases: list[list[str]],
    ) -> TextCCInputs:
        full_token_indices: list[list[int]] = []
        iso_token_indices: list[list[int]] = []
        iso_texts: list[str] = []
        sample_indices: list[int] = []
        for sample_idx, (caption, phrases) in enumerate(zip(captions, noun_phrases)):
            for phrase, full_idx, iso_idx in self._iter_valid_phrase_alignments(caption, phrases):
                full_token_indices.append(full_idx)
                iso_token_indices.append(iso_idx)
                iso_texts.append(phrase)
                sample_indices.append(sample_idx)
        return TextCCInputs(full_token_indices, iso_token_indices, iso_texts, sample_indices)

    def _iter_valid_phrase_alignments(self, caption: str, phrases: list[str]):
        clipped_phrases = list(phrases or [])[: self.cfg.loss.max_noun_phrases]
        aligned = align_noun_phrases(caption, clipped_phrases, self.model.tokenizer)
        for phrase, full_idx in zip(clipped_phrases, aligned):
            if not full_idx:
                continue
            iso_idx = align_single_phrase(phrase, phrase, self.model.tokenizer)
            if iso_idx:
                yield phrase, full_idx, iso_idx

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
        losses = self._compute_losses(batch, include_query_negative=False, include_text_cc=False)
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
        metrics.update(self._parameter_metrics())
        self._log_direct_metrics(metrics)

    def _parameter_metrics(self) -> dict[str, float]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        frozen = sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)
        trainable_encoder = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("model.encoders.")
        )
        trainable_head = trainable - trainable_encoder
        return {
            "num_trainable_params": float(trainable),
            "num_trainable_encoder_params": float(trainable_encoder),
            "num_trainable_head_params": float(trainable_head),
            "num_frozen_params": float(frozen),
            "num_total_params": float(trainable + frozen),
            "freeze_vision_encoder": float(bool(self.cfg.model.freeze_vision_encoder)),
            "freeze_text_encoder": float(bool(self.cfg.model.freeze_text_encoder)),
            "trainable_vision_layers": float(self.cfg.model.trainable_vision_layers),
            "trainable_text_layers": float(self.cfg.model.trainable_text_layers),
            "use_gradient_checkpointing": float(bool(self.cfg.model.use_gradient_checkpointing)),
            "head_learning_rate": float(self.cfg.train.learning_rate),
            "encoder_learning_rate": float(self.cfg.train.encoder_learning_rate or self.cfg.train.learning_rate),
            "query_negative_weight": float(self.cfg.loss.query_negative_weight),
        }

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
        head_named_parameters = []
        encoder_named_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("model.encoders."):
                encoder_named_parameters.append((name, parameter))
            else:
                head_named_parameters.append((name, parameter))

        if not head_named_parameters and not encoder_named_parameters:
            raise RuntimeError("No trainable parameters. Check model freeze settings.")
        parameter_groups = []
        self._append_optimizer_groups(
            parameter_groups,
            head_named_parameters,
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
            name="head",
        )
        self._append_optimizer_groups(
            parameter_groups,
            encoder_named_parameters,
            lr=self.cfg.train.encoder_learning_rate or self.cfg.train.learning_rate,
            weight_decay=(
                self.cfg.train.encoder_weight_decay
                if self.cfg.train.encoder_weight_decay is not None
                else self.cfg.train.weight_decay
            ),
            name="encoder",
        )
        optimizer = AdamW(
            parameter_groups,
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

    def _append_optimizer_groups(
        self,
        parameter_groups: list[dict[str, Any]],
        named_parameters: list[tuple[str, torch.nn.Parameter]],
        lr: float,
        weight_decay: float,
        name: str,
    ) -> None:
        if not named_parameters:
            return
        decay_parameters = []
        no_decay_parameters = []
        for parameter_name, parameter in named_parameters:
            if self._should_skip_weight_decay(parameter_name, parameter):
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        if decay_parameters:
            parameter_groups.append(
                {"params": decay_parameters, "lr": lr, "weight_decay": weight_decay, "name": f"{name}_decay"}
            )
        if no_decay_parameters:
            parameter_groups.append(
                {"params": no_decay_parameters, "lr": lr, "weight_decay": 0.0, "name": f"{name}_no_decay"}
            )

    def _should_skip_weight_decay(self, parameter_name: str, parameter: torch.nn.Parameter) -> bool:
        lower_name = parameter_name.lower()
        return (
            parameter.ndim < 2
            or lower_name.endswith(".bias")
            or "norm" in lower_name
            or "embedding" in lower_name
        )


def load_cir_module_from_checkpoint(
    checkpoint_path: str,
    cfg: AppConfig,
    map_location: torch.device | str | None = None,
) -> CIRLightningModule:
    """Load a training checkpoint without requiring Lightning at inference time."""

    if hasattr(CIRLightningModule, "load_from_checkpoint"):
        return CIRLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg, map_location=map_location)

    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint does not contain a state_dict: {checkpoint_path}")

    module = CIRLightningModule(cfg)
    module.load_state_dict(state_dict)
    return module


def _torch_load_checkpoint(
    checkpoint_path: str,
    map_location: torch.device | str | None,
) -> Any:
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)
