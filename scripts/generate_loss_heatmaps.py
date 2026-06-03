from __future__ import annotations

import argparse
import base64
import html
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageOps

from cir_pipeline.config import AppConfig, load_config
from cir_pipeline.data.cirr_dataset import CIRRDataset
from cir_pipeline.data.transforms import build_clip_transform
from cir_pipeline.engine.train_module import load_cir_module_from_checkpoint
from cir_pipeline.models.losses import text_concept_consistency_loss
from cir_pipeline.text.noun_phrase_parser import NounPhraseParser
from cir_pipeline.text.token_alignment import align_noun_phrases, align_single_phrase
from cir_pipeline.utils.io import ensure_dir, write_json
from cir_pipeline.utils.serve import resolve_server_urls


MODEL_VARIANTS = {
    "cl": {
        "label": "Contrastive loss",
        "default_checkpoint": "checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt",
        "loss_kind": "contrastive",
    },
    "cl_cc": {
        "label": "CL + CC",
        "default_checkpoint": "checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt",
        "loss_kind": "combined",
    },
}


@dataclass
class ModelSlot:
    key: str
    label: str
    checkpoint: Path | None
    loss_kind: str
    model: Any | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.error is None


@dataclass
class HeatmapState:
    cfg: AppConfig
    split: str
    device: torch.device
    dataset: CIRRDataset
    samples_by_pair_id: dict[str, dict[str, Any]]
    noun_parser: NounPhraseParser
    model_slots: dict[str, ModelSlot]
    output_dir: Path
    report_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a heatmap UI for CIRR Text-CC and contrastive ablations.")
    parser.add_argument("--config", default="configs/cirr_context_cir_cl_cc.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test1"])
    parser.add_argument(
        "--cl_checkpoint",
        default=MODEL_VARIANTS["cl"]["default_checkpoint"],
        help="Checkpoint for the CL-only model slot.",
    )
    parser.add_argument(
        "--cl_cc_checkpoint",
        default=MODEL_VARIANTS["cl_cc"]["default_checkpoint"],
        help="Checkpoint for the CL+CC model slot.",
    )
    parser.add_argument("--limit", type=int, default=300, help="Number of samples shown in the selector.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host. Use 0.0.0.0 to open the UI to other machines.")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def _load_model_slot(
    key: str,
    cfg: AppConfig,
    checkpoint: Path | None,
    device: torch.device,
) -> ModelSlot:
    spec = MODEL_VARIANTS[key]
    slot = ModelSlot(
        key=key,
        label=str(spec["label"]),
        checkpoint=checkpoint,
        loss_kind=str(spec["loss_kind"]),
    )
    if checkpoint is None:
        slot.error = "No checkpoint configured"
        return slot
    if not checkpoint.exists():
        slot.error = f"Checkpoint not found: {checkpoint}"
        return slot
    try:
        module = load_cir_module_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device)
        slot.model = module.model.to(device).eval()
    except Exception as exc:
        slot.error = str(exc)
    return slot


def _create_state(
    config_path: str | Path,
    split: str,
    checkpoints: dict[str, str | None],
    limit: int,
) -> HeatmapState:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CIRRDataset(cfg.data, split, "relative", transform=None, noun_phrase_parser=None)
    samples = dataset.samples[: max(1, limit)]
    samples_by_pair_id = {str(sample["pair_id"]): sample for sample in samples}
    noun_parser = NounPhraseParser(
        max_phrases=max(1, cfg.loss.max_noun_phrases),
      backend=cfg.data.noun_phrase_backend if cfg.data.noun_phrase_backend != "none" else "regex",
      stanza_resources_dir=cfg.data.stanza_resources_dir,
      stanza_download_method=cfg.data.stanza_download_method,
      stanza_np_extractor=getattr(cfg.data, "noun_phrase_extractor", "constituency"),
    )
    model_slots = {
        key: _load_model_slot(key, cfg, _resolve_project_path(checkpoints.get(key)), device)
        for key in MODEL_VARIANTS
    }
    output_dir = ensure_dir(Path(cfg.project.output_dir) / "heatmaps")
    report_dir = ensure_dir(output_dir / "reports")
    return HeatmapState(
        cfg=cfg,
        split=split,
        device=device,
        dataset=dataset,
        samples_by_pair_id=samples_by_pair_id,
        noun_parser=noun_parser,
        model_slots=model_slots,
        output_dir=output_dir,
        report_dir=report_dir,
    )


def _sample_options(state: HeatmapState, selected_pair_id: str | None = None) -> str:
    rows = []
    for pair_id, sample in state.samples_by_pair_id.items():
        caption = str(sample.get("caption", ""))
        reference = str(sample.get("reference_name", ""))
        target = str(sample.get("target_name") or "")
        selected = " selected" if selected_pair_id and pair_id == selected_pair_id else ""
        target_html = (
            f'<img src="/artifact/{quote(_relative_to_root(state.dataset.resolve_image_path(target)), safe="/")}" alt="{html.escape(target)}">'
            if target
            else '<div class="sample-empty">no target</div>'
        )
        rows.append(
            f"""<label class="sample-card{selected}">
  <input type="radio" name="pair_id" value="{html.escape(pair_id)}"{" checked" if selected else ""}>
  <div class="sample-thumbs">
    <img src="/artifact/{quote(_relative_to_root(state.dataset.resolve_image_path(reference)), safe="/")}" alt="{html.escape(reference)}">
    {target_html}
  </div>
  <div class="sample-text">
    <div class="sample-id">#{html.escape(pair_id)}</div>
    <div class="sample-caption">{html.escape(caption)}</div>
  </div>
</label>"""
        )
    return "\n".join(rows)


def _variant_options(state: HeatmapState, selected_variant: str = "cl_cc") -> str:
    cards = []
    for key, slot in state.model_slots.items():
        checked = " checked" if key == selected_variant else ""
        status = "ready" if slot.available else "missing"
        checkpoint = str(slot.checkpoint) if slot.checkpoint else "none"
        cards.append(
            f"""
            <label class="variant {status}">
              <input type="radio" name="variant" value="{key}"{checked}>
              <span class="variant-main">{html.escape(slot.label)}</span>
              <span class="variant-sub">{html.escape(status)} &middot; {html.escape(checkpoint)}</span>
            </label>
            """
        )
    return "\n".join(cards)


def _image_to_data_url(path: Path) -> str:
    content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _page(
    state: HeatmapState,
    selected_pair_id: str | None = None,
    selected_variant: str = "cl_cc",
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    result = result or {}
    sample = result.get("sample") or {}
    phrases = result.get("phrases") or []
    heatmaps = result.get("heatmaps") or []
    report_json = result.get("report_json")
    original_url = result.get("reference_url")

    heatmap_cards = "\n".join(_heatmap_card(row) for row in heatmaps)
    phrase_tags = " ".join(f"<span>{html.escape(str(phrase))}</span>" for phrase in phrases)
    if not phrase_tags:
        phrase_tags = '<span class="empty-tag">No phrases</span>'
    preview = (
        f'<img src="{original_url}" alt="reference image">'
        if original_url
        else '<div class="preview-empty">Select a sample</div>'
    )
    alert = f'<div class="alert">{html.escape(error)}</div>' if error else ""
    unavailable = result.get("unavailable")
    unavailable_html = (
        f'<div class="alert">{html.escape(str(unavailable))}</div>'
        if unavailable
        else ""
    )
    report_link = (
        f'<a href="/artifact/{quote(str(report_json), safe="/")}">heatmap report</a>'
        if report_json
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ConText-CIRR | Loss Heatmaps</title>
  <meta name="description" content="Interactive heatmap viewer for contrastive and CL+CC phrase attention analysis in the ConText-CIRR workspace.">
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #fff;
      --line: #d7dde7;
      --text: #18212f;
      --muted: #667085;
      --accent: #0f766e;
      --danger: #b42318;
      --warn: #b54708;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    .bar {{
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 14px 0;
      display: flex;
      justify-content: space-between;
      gap: 16px;
    }}
    .brand {{ display: grid; gap: 4px; }}
    h1 {{
      margin: 0;
      font-size: 18px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      width: min(1280px, calc(100vw - 32px));
      margin: 18px auto 34px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    form {{
      padding: 14px;
      display: grid;
      gap: 12px;
    }}
    .field {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 700;
      color: #344054;
    }}
    select, input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      font-size: 14px;
      background: #fff;
    }}
    .sample-grid {{
      max-height: 430px;
      overflow: auto;
      display: grid;
      gap: 8px;
      padding-right: 2px;
    }}
    .sample-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      display: grid;
      grid-template-columns: 18px 112px 1fr;
      gap: 8px;
      cursor: pointer;
      background: #fff;
      min-width: 0;
    }}
    .sample-card.selected {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.12);
    }}
    .sample-thumbs {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}
    .sample-thumbs img, .sample-empty {{
      width: 52px;
      height: 52px;
      border-radius: 6px;
      object-fit: cover;
      background: #eef2f6;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 10px;
      text-align: center;
    }}
    .sample-text {{
      min-width: 0;
      display: grid;
      gap: 4px;
      align-content: start;
    }}
    .sample-id {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .sample-caption {{
      font-size: 12px;
      line-height: 1.3;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .variant-grid {{
      display: grid;
      gap: 8px;
    }}
    .variant {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      display: grid;
      grid-template-columns: 18px 1fr;
      column-gap: 8px;
      cursor: pointer;
    }}
    .variant-main {{
      font-weight: 700;
      font-size: 13px;
    }}
    .variant-sub {{
      grid-column: 2;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .variant.missing .variant-sub {{ color: var(--warn); }}
    button {{
      height: 40px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }}
    .preview {{
      aspect-ratio: 1 / 1;
      background: #eef2f6;
      display: grid;
      place-items: center;
      border-bottom: 1px solid var(--line);
    }}
    .preview img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .preview-empty {{
      color: var(--muted);
      font-size: 13px;
    }}
    .sample-meta {{
      padding: 12px 14px;
      display: grid;
      gap: 6px;
      border-bottom: 1px solid var(--line);
    }}
    .caption {{
      font-size: 14px;
      line-height: 1.35;
    }}
    .links {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 13px;
    }}
    .links a {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    .phrases {{
      padding: 12px 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
    }}
    .phrases span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      background: #f8fafc;
    }}
    .alert {{
      margin: 12px 14px;
      color: var(--danger);
      font-size: 13px;
      font-weight: 700;
    }}
    .grid {{
      padding: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
      gap: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: block;
    }}
    .card-title {{
      padding: 8px;
      min-height: 42px;
      font-size: 12px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      .bar {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div class="brand">
        <div class="sample-id">ConText-CIRR Workspace</div>
        <h1>Loss Heatmaps</h1>
      </div>
      <div class="muted">split={html.escape(state.split)} &middot; device={state.device.type} &middot; samples={len(state.samples_by_pair_id)}</div>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="preview">{preview}</div>
      <form method="post" action="/generate">
        <div class="field">Sample</div>
        <div class="sample-grid">{_sample_options(state, selected_pair_id)}</div>
        <div class="variant-grid">{_variant_options(state, selected_variant)}</div>
        <label class="field">Max phrases
          <input name="max_phrases" type="number" min="1" max="12" value="{html.escape(str(result.get("max_phrases", state.cfg.loss.max_noun_phrases or 5)))}">
        </label>
        <button type="submit">Generate heatmaps</button>
      </form>
    </section>
    <section class="panel">
      <div class="sample-meta">
        <div class="muted">pair_id={html.escape(str(sample.get("pair_id", "")))} &middot; reference={html.escape(str(sample.get("reference_name", "")))} &middot; target={html.escape(str(sample.get("target_name", "")))}</div>
        <div class="caption">{html.escape(str(sample.get("caption", "")))}</div>
        <div class="links">{report_link}</div>
      </div>
      <div class="phrases">{phrase_tags}</div>
      {alert}
      {unavailable_html}
      <div class="grid">{heatmap_cards}</div>
    </section>
  </main>
</body>
</html>"""


def _heatmap_card(row: dict[str, Any]) -> str:
    url = f"/artifact/{quote(str(row['relative_path']), safe='/')}"
    title = str(row.get("title", row.get("phrase", "")))
    return f"""<article class="card">
  <img src="{url}" alt="{html.escape(title)}">
  <div class="card-title">{html.escape(title)}</div>
</article>"""


def _relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _clip_image_for_overlay(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (224, 224), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))


def _save_reference_preview(image_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "reference.png"
    _clip_image_for_overlay(image_path).save(out)
    return out


def _to_heat_grid(values: torch.Tensor) -> np.ndarray:
    array = values.detach().float().cpu().numpy()
    if array.size > 1:
        array = array[1:]
    side = int(np.sqrt(array.size))
    if side * side != array.size:
        side = int(np.floor(np.sqrt(array.size)))
        array = array[: side * side]
    if side <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    grid = array.reshape(side, side).astype(np.float32)
    grid -= float(grid.min())
    denom = float(grid.max())
    if denom > 1e-8:
        grid /= denom
    return grid


def _colorize_heatmap(grid: np.ndarray) -> Image.Image:
    x = np.clip(grid, 0.0, 1.0)
    red = np.clip(1.6 * x, 0.0, 1.0)
    green = np.clip(1.8 - np.abs(2.0 * x - 1.0) * 1.8, 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - x), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BICUBIC)


def _save_overlay(base: Image.Image, values: torch.Tensor, path: Path) -> None:
    grid = _to_heat_grid(values)
    heat = _colorize_heatmap(grid)
    overlay = Image.blend(base, heat, alpha=0.48)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path)


def _average_attention(attention_maps: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(attention_maps, dim=0).mean(dim=(0, 2))


def _average_attribution(attention_maps: list[torch.Tensor]) -> torch.Tensor:
    per_layer = []
    for attn in attention_maps:
        grad = attn.grad
        if grad is None:
            per_layer.append(attn.detach().clamp_min(0))
        else:
            per_layer.append((attn.detach() * grad.detach().clamp_min(0)).clamp_min(0))
    score = torch.stack(per_layer, dim=0).mean(dim=(0, 2))
    if float(score.sum()) <= 1e-8:
        score = _average_attention(attention_maps)
    return score


def _phrase_heat_values(score: torch.Tensor, token_indices: list[int]) -> torch.Tensor:
    valid = [idx for idx in token_indices if 0 <= idx < score.shape[-1]]
    if not valid:
        return torch.zeros(score.shape[0], device=score.device)
    return score[:, valid].mean(dim=-1)


def _retain_attention_grads(attention_maps: list[torch.Tensor]) -> None:
    for attn in attention_maps:
        if attn.requires_grad:
            attn.retain_grad()


def _compute_loss_for_variant(
    slot: ModelSlot,
    query_out: dict[str, Any],
    iso_out: dict[str, Any] | None,
    target_tensor: torch.Tensor | None,
    full_token_indices: list[list[int]],
    iso_token_indices: list[list[int]],
) -> torch.Tensor:
    embedding = query_out["embeddings"]
    loss = embedding.sum() * 0.0
    if slot.loss_kind in {"contrastive", "combined"} and target_tensor is not None:
        target_embedding = slot.model.encode_gallery(target_tensor)
        loss = loss - (embedding * target_embedding).sum(dim=-1).mean()
    if slot.loss_kind in {"cc", "combined"} and iso_out is not None and full_token_indices:
        sample_indices = [0 for _ in full_token_indices]
        cc_loss = text_concept_consistency_loss(
            query_out["attention_maps"],
            iso_out["attention_maps"],
            full_token_indices,
            iso_token_indices,
            sample_indices,
            epsilon=0.0,
        )
        lambda_cc = slot.model.cfg.dropout * 0.0 + 1.0
        if slot.loss_kind == "combined":
            lambda_cc = max(float(getattr(slot.model.cfg, "dropout", 0.0)) * 0.0 + 1.0, 1.0)
        loss = loss + lambda_cc * cc_loss
    return loss


def _generate_heatmaps(
    state: HeatmapState,
    sample: dict[str, Any],
    phrases: list[str],
    slot: ModelSlot,
    max_phrases: int,
) -> dict[str, Any]:
    if not slot.available:
        return {"unavailable": slot.error or "Model unavailable", "heatmaps": []}

    model = slot.model
    image_path = state.dataset.resolve_image_path(str(sample["reference_name"]))
    target_path = (
        state.dataset.resolve_image_path(str(sample["target_name"]))
        if sample.get("target_name")
        else None
    )
    pair_id = str(sample["pair_id"])
    out_dir = ensure_dir(state.output_dir / slot.key / f"pair_{pair_id}")
    base = _clip_image_for_overlay(image_path)
    reference_out = _save_reference_preview(image_path, out_dir)

    selected_phrases = phrases[:max_phrases]
    full_token_indices = align_noun_phrases(str(sample["caption"]), selected_phrases, model.tokenizer)
    valid_rows = [
        (phrase, full_idx, align_single_phrase(phrase, phrase, model.tokenizer))
        for phrase, full_idx in zip(selected_phrases, full_token_indices)
        if full_idx
    ]
    valid_rows = [(phrase, full_idx, iso_idx) for phrase, full_idx, iso_idx in valid_rows if iso_idx]

    transform = build_clip_transform(
        train=False,
        backbone=state.cfg.model.backbone,
        transform_name=getattr(state.cfg.data, "image_transform", "clip"),
        target_pad_ratio=float(getattr(state.cfg.data, "target_pad_ratio", 1.25)),
    )
    image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(state.device)
    target_tensor = (
        transform(Image.open(target_path).convert("RGB")).unsqueeze(0).to(state.device)
        if target_path is not None
        else None
    )
    for parameter in model.parameters():
        parameter.grad = None
    with torch.enable_grad():
        query_out = model.forward_with_attention(image_tensor, [str(sample["caption"])])
        _retain_attention_grads(query_out["attention_maps"])
        iso_out = None
        if valid_rows:
            iso_images = image_tensor.repeat(len(valid_rows), 1, 1, 1)
            iso_texts = [row[0] for row in valid_rows]
            iso_out = model.forward_with_attention(iso_images, iso_texts)
            _retain_attention_grads(iso_out["attention_maps"])
        loss = _compute_loss_for_variant(
            slot,
            query_out,
            iso_out,
            target_tensor,
            [row[1] for row in valid_rows],
            [row[2] for row in valid_rows],
        )
        loss.backward()

    full_score = _average_attribution(query_out["attention_maps"])[0]
    iso_score = _average_attribution(iso_out["attention_maps"]) if iso_out is not None else None
    heatmaps = [
        {
            "title": "reference",
            "phrase": None,
            "relative_path": _relative_to_root(reference_out),
        }
    ]
    for idx, (phrase, full_idx, _iso_idx) in enumerate(valid_rows):
        full_path = out_dir / f"full_{idx:02d}.png"
        _save_overlay(base, _phrase_heat_values(full_score, full_idx), full_path)
        heatmaps.append(
            {
                "title": f'full: "{phrase}"',
                "phrase": phrase,
                "kind": "full_caption",
                "relative_path": _relative_to_root(full_path),
            }
        )
        if iso_score is not None:
            iso_path = out_dir / f"isolated_{idx:02d}.png"
            _save_overlay(base, _phrase_heat_values(iso_score[idx], _iso_idx), iso_path)
            heatmaps.append(
                {
                    "title": f'isolated: "{phrase}"',
                    "phrase": phrase,
                    "kind": "isolated_phrase",
                    "relative_path": _relative_to_root(iso_path),
                }
            )
    return {
        "loss_value": float(loss.detach().cpu()),
        "heatmaps": heatmaps,
    }


def _generate_result(
    state: HeatmapState,
    pair_id: str,
    variant: str,
    max_phrases: int,
) -> dict[str, Any]:
    if pair_id not in state.samples_by_pair_id:
        raise KeyError(f"Unknown pair_id: {pair_id}")
    if variant not in state.model_slots:
        raise KeyError(f"Unknown model variant: {variant}")
    sample = dict(state.samples_by_pair_id[pair_id])
    phrases = state.noun_parser.extract(str(sample["caption"]))[:max_phrases]
    slot = state.model_slots[variant]
    model_result = _generate_heatmaps(state, sample, phrases, slot, max_phrases)
    image_path = state.dataset.resolve_image_path(str(sample["reference_name"]))
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": state.split,
        "sample": sample,
        "phrases": phrases,
        "noun_phrase_backend": state.noun_parser.backend,
        "max_phrases": max_phrases,
        "variant": variant,
        "variant_label": slot.label,
        "reference_image_path": _relative_to_root(image_path),
        **model_result,
    }
    report_path = state.report_dir / f"heatmap_report_{variant}.json"
    write_json(result, report_path)
    result["report_json"] = _relative_to_root(report_path)
    result["reference_url"] = _image_to_data_url(image_path)
    return result


def create_app(
    config_path: str | Path,
    split: str,
    checkpoints: dict[str, str | None],
    limit: int,
) -> FastAPI:
    state = _create_state(config_path, split, checkpoints, limit)
    app = FastAPI(title="CIRR Loss Heatmaps")
    app.state.heatmaps = state

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        first_pair_id = next(iter(state.samples_by_pair_id), None)
        return _page(state, selected_pair_id=first_pair_id)

    @app.post("/generate", response_class=HTMLResponse)
    def generate_page(
        pair_id: Annotated[str, Form(...)],
        variant: Annotated[str, Form()] = "cl_cc",
        max_phrases: Annotated[int, Form()] = 5,
    ) -> str:
        max_phrases = min(max(int(max_phrases), 1), 12)
        try:
            result = _generate_result(state, pair_id, variant, max_phrases)
            return _page(state, pair_id, variant, result)
        except Exception as exc:
            return _page(state, pair_id, variant, error=str(exc))

    @app.get("/api/generate")
    def generate_api(
        pair_id: Annotated[str, Query(...)],
        variant: Annotated[str, Query()] = "cl_cc",
        max_phrases: Annotated[int, Query()] = 5,
    ) -> JSONResponse:
        max_phrases = min(max(int(max_phrases), 1), 12)
        return JSONResponse(_generate_result(state, pair_id, variant, max_phrases))

    @app.get(
        "/artifact/{path:path}",
        responses={
            403: {"description": "Forbidden"},
            404: {"description": "Not found"},
        },
    )
    def artifact(path: str) -> FileResponse:
        resolved = (ROOT / path).resolve()
        allowed_roots = [ROOT / "outputs", ROOT / "CIRR"]
        if not any(resolved.is_relative_to(root.resolve()) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="Forbidden artifact path")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
        return FileResponse(resolved)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device": state.device.type,
            "split": state.split,
            "models": {
                key: {
                    "label": slot.label,
                    "available": slot.available,
                    "checkpoint": str(slot.checkpoint) if slot.checkpoint else None,
                    "error": slot.error,
                }
                for key, slot in state.model_slots.items()
            },
        }

    return app


def main() -> None:
    args = parse_args()
    checkpoints = {
        "cl": args.cl_checkpoint,
        "cl_cc": args.cl_cc_checkpoint,
    }
    app = create_app(args.config, args.split, checkpoints, args.limit)
    urls = resolve_server_urls(args.host, args.port)
    print(f"Heatmap UI (local): {urls.local_url}")
    if urls.lan_url:
        print(f"Heatmap UI (LAN):   {urls.lan_url}")
    elif args.host not in {"127.0.0.1", "localhost"}:
        print(f"Heatmap UI (bind):  {urls.bind_url}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
