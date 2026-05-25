from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from cir_pipeline.config import AppConfig, load_config
from cir_pipeline.data.cirr_dataset import CIRRDataset


@dataclass
class DatasetState:
    cfg: AppConfig
    datasets: dict[str, CIRRDataset]
    limit: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CIRR dataset visualization UI.")
    parser.add_argument("--config", default="configs/cirr_context_cir.yaml")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test1"], default=["train", "val", "test1"])
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args()


def _create_state(config_path: str | Path, splits: list[str], limit: int) -> DatasetState:
    cfg = load_config(config_path)
    datasets = {
        split: CIRRDataset(cfg.data, split, "relative", transform=None, noun_phrase_parser=None)
        for split in splits
    }
    return DatasetState(cfg=cfg, datasets=datasets, limit=max(1, limit))


def _image_url(split: str, image_name: str | None) -> str:
    if not image_name:
        return ""
    return f"/image/{quote(split, safe='')}/{quote(image_name, safe='')}"


def _split_options(state: DatasetState, selected_split: str) -> str:
    rows = []
    for split, dataset in state.datasets.items():
        selected = " selected" if split == selected_split else ""
        rows.append(
            f'<option value="{html.escape(split)}"{selected}>{html.escape(split)} ({len(dataset):,})</option>'
        )
    return "\n".join(rows)


def _find_sample(dataset: CIRRDataset, pair_id: str | None) -> dict[str, Any]:
    if pair_id:
        for sample in dataset.samples:
            if str(sample.get("pair_id")) == str(pair_id):
                return dict(sample)
    return dict(dataset.samples[0]) if dataset.samples else {}


def _sample_cards(dataset: CIRRDataset, split: str, selected_pair_id: str | None, limit: int) -> str:
    cards = []
    for sample in dataset.samples[:limit]:
        pair_id = str(sample.get("pair_id"))
        selected = " selected" if pair_id == str(selected_pair_id) else ""
        ref_name = str(sample.get("reference_name") or "")
        target_name = str(sample.get("target_name") or "")
        target = (
            f'<img src="{_image_url(split, target_name)}" alt="{html.escape(target_name)}">'
            if target_name and target_name != "None"
            else '<div class="thumb-empty">no target</div>'
        )
        cards.append(
            f"""<a class="sample-card{selected}" href="/?split={quote(split)}&pair_id={quote(pair_id)}">
  <div class="thumbs">
    <img src="{_image_url(split, ref_name)}" alt="{html.escape(ref_name)}">
    {target}
  </div>
  <div class="card-meta">
    <div class="pair">#{html.escape(pair_id)}</div>
    <div class="caption">{html.escape(str(sample.get("caption", "")))}</div>
  </div>
</a>"""
        )
    return "\n".join(cards)


def _detail(dataset: CIRRDataset, split: str, sample: dict[str, Any]) -> str:
    if not sample:
        return '<div class="empty">No samples</div>'
    pair_id = str(sample.get("pair_id", ""))
    ref_name = str(sample.get("reference_name") or "")
    target_name = sample.get("target_name")
    target_name = str(target_name) if target_name else ""
    target_panel = (
        f"""<figure>
  <img src="{_image_url(split, target_name)}" alt="{html.escape(target_name)}">
  <figcaption>{html.escape(target_name)}</figcaption>
</figure>"""
        if target_name
        else """<figure>
  <div class="missing-target">No target in this split</div>
  <figcaption>target</figcaption>
</figure>"""
    )
    subset = ", ".join(str(item) for item in sample.get("img_set", [])[:6])
    return f"""<div class="detail-head">
  <div>
    <div class="eyebrow">pair_id {html.escape(pair_id)} &middot; split {html.escape(split)}</div>
    <h2>{html.escape(str(sample.get("caption", "")))}</h2>
  </div>
</div>
<div class="flow">
  <figure>
    <img src="{_image_url(split, ref_name)}" alt="{html.escape(ref_name)}">
    <figcaption>{html.escape(ref_name)}</figcaption>
  </figure>
  <div class="caption-box">{html.escape(str(sample.get("caption", "")))}</div>
  {target_panel}
</div>
<div class="facts">
  <span>img_set_id: {html.escape(str(sample.get("img_set_id", "")))}</span>
  <span>subset: {html.escape(subset)}</span>
</div>"""


def _page(state: DatasetState, split: str, pair_id: str | None) -> str:
    if split not in state.datasets:
        split = next(iter(state.datasets))
    dataset = state.datasets[split]
    sample = _find_sample(dataset, pair_id)
    selected_pair_id = str(sample.get("pair_id", "")) if sample else None
    cards = _sample_cards(dataset, split, selected_pair_id, state.limit)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CIRR Dataset Viewer</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #fff;
      --line: #d7dde7;
      --text: #17202a;
      --muted: #667085;
      --accent: #0f766e;
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
      width: min(1320px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 14px 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{ margin: 0; font-size: 18px; }}
    h2 {{
      margin: 4px 0 0;
      font-size: 19px;
      line-height: 1.3;
    }}
    form {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    select, button {{
      height: 36px;
      border-radius: 6px;
      font: inherit;
      font-size: 14px;
    }}
    select {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 0 10px;
    }}
    button {{
      border: 0;
      background: var(--accent);
      color: #fff;
      padding: 0 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    main {{
      width: min(1320px, calc(100vw - 32px));
      margin: 18px auto 34px;
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .list {{
      max-height: calc(100vh - 112px);
      overflow: auto;
      padding: 12px;
      display: grid;
      gap: 10px;
    }}
    .sample-card {{
      color: inherit;
      text-decoration: none;
      display: grid;
      grid-template-columns: 126px 1fr;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }}
    .sample-card.selected {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.12);
    }}
    .thumbs {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}
    .thumbs img, .thumb-empty {{
      width: 60px;
      height: 60px;
      object-fit: cover;
      border-radius: 6px;
      background: #eef2f6;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 10px;
      text-align: center;
    }}
    .card-meta {{
      min-width: 0;
      display: grid;
      gap: 4px;
      align-content: start;
    }}
    .pair {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .caption {{
      font-size: 13px;
      line-height: 1.3;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .detail {{
      padding: 16px;
      display: grid;
      gap: 16px;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .flow {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(190px, 0.82fr) minmax(180px, 1fr);
      align-items: center;
      gap: 14px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    figure img, .missing-target {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: grid;
      place-items: center;
      background: #eef2f6;
      color: var(--muted);
      font-size: 13px;
    }}
    figcaption {{
      min-height: 36px;
      padding: 9px;
      font-size: 12px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .caption-box {{
      min-height: 96px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      padding: 14px;
      display: grid;
      place-items: center;
      text-align: center;
      line-height: 1.35;
      font-weight: 700;
    }}
    .facts {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }}
    .facts span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: #f8fafc;
    }}
    .empty {{
      padding: 48px 16px;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 940px) {{
      .bar {{ flex-direction: column; align-items: flex-start; }}
      main {{ grid-template-columns: 1fr; }}
      .list {{ max-height: none; }}
      .flow {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <h1>CIRR Dataset Viewer</h1>
      <form method="get" action="/">
        <select name="split">{_split_options(state, split)}</select>
        <button type="submit">Open</button>
      </form>
    </div>
  </header>
  <main>
    <section class="panel list">{cards}</section>
    <section class="panel detail">{_detail(dataset, split, sample)}</section>
  </main>
</body>
</html>"""


def create_app(config_path: str | Path, splits: list[str], limit: int) -> FastAPI:
    state = _create_state(config_path, splits, limit)
    app = FastAPI(title="CIRR Dataset Viewer")
    app.state.dataset_viewer = state

    @app.get("/", response_class=HTMLResponse)
    def index(split: str = Query("val"), pair_id: str | None = Query(None)) -> str:
        return _page(state, split, pair_id)

    @app.get("/api/sample")
    def sample_api(split: str = Query("val"), pair_id: str | None = Query(None)) -> JSONResponse:
        if split not in state.datasets:
            raise HTTPException(status_code=404, detail=f"Unknown split: {split}")
        return JSONResponse(_find_sample(state.datasets[split], pair_id))

    @app.get("/image/{split}/{image_name}")
    def image(split: str, image_name: str) -> FileResponse:
        dataset = state.datasets.get(split)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Unknown split: {split}")
        try:
            path = dataset.resolve_image_path(image_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_name}")
        return FileResponse(path)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "splits": {split: len(dataset) for split, dataset in state.datasets.items()},
        }

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args.config, args.splits, args.limit)
    print(f"Dataset UI: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
