from __future__ import annotations

import argparse
import base64
import html
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from cir_pipeline.config import AppConfig, load_config
from cir_pipeline.engine.train_module import load_cir_module_from_checkpoint
from cir_pipeline.retrieval.faiss_index import FaissIndex, load_mapping
from cir_pipeline.retrieval.search import encode_query_image_text, search_loaded_index


@dataclass
class IndexSet:
    split: str
    index_path: Path
    mapping_path: Path
    faiss_index: FaissIndex
    mapping: list[dict[str, Any]]
    gallery_paths: dict[str, Path]


@dataclass
class DemoState:
    cfg: AppConfig
    checkpoint: Path
    upload_dir: Path
    device: torch.device
    model: Any
    index_sets: dict[str, IndexSet]
    index_errors: dict[str, str]
    lock: threading.Lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a FastAPI demo for CIR image + text retrieval.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test1"])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test1"],
        default=None,
        help="FAISS gallery splits to load. Defaults to val and test1 when --index is not used.",
    )
    parser.add_argument("--index", default=None)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def _default_index_path(cfg: AppConfig, split: str) -> Path:
    return _resolve_project_path(Path(cfg.faiss.output_dir) / f"cirr_{split}" / "index.faiss")


def _default_mapping_path(cfg: AppConfig, split: str) -> Path:
    return _resolve_project_path(Path(cfg.faiss.output_dir) / f"cirr_{split}" / "mapping.json")


def _requested_splits(split: str, splits: list[str] | None, index_path: str | Path | None) -> list[str]:
    requested = [split] if index_path else (splits or ["val", "test1"])
    unique: list[str] = []
    for item in requested:
        if item not in unique:
            unique.append(item)
    return unique


def _load_gallery_paths(mapping: list[dict[str, Any]]) -> dict[str, Path]:
    gallery_paths: dict[str, Path] = {}
    for item in mapping:
        image_name = str(item["image_name"])
        gallery_paths[image_name] = _resolve_project_path(item["image_path"])
    return gallery_paths


def _validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_index_set(
    cfg: AppConfig,
    split: str,
    index_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> IndexSet:
    index = _resolve_project_path(index_path) if index_path else _default_index_path(cfg, split)
    mapping = _resolve_project_path(mapping_path) if mapping_path else _default_mapping_path(cfg, split)
    _validate_file(index, f"FAISS index for {split}")
    _validate_file(mapping, f"FAISS mapping for {split}")
    loaded_mapping = load_mapping(mapping)
    return IndexSet(
        split=split,
        index_path=index,
        mapping_path=mapping,
        faiss_index=FaissIndex.load(index),
        mapping=loaded_mapping,
        gallery_paths=_load_gallery_paths(loaded_mapping),
    )


def _image_data_url(path: Path) -> str:
    content_types = {
        ".bmp": "image/bmp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    content_type = content_types.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _save_upload(upload_dir: Path, upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".png"
    path = upload_dir / f"{uuid4().hex}{suffix}"
    with path.open("wb") as handle:
        handle.write(upload.file.read())
    return path


def _page(
    *,
    state: DemoState,
    results: list[dict[str, Any]] | None = None,
    query_preview: str | None = None,
    text: str = "",
    reference_name: str = "",
    gallery_split: str = "val",
    top_k: int = 10,
    error: str | None = None,
) -> str:
    results = results or []
    gallery_count = sum(len(index_set.gallery_paths) for index_set in state.index_sets.values())
    result_cards = "\n".join(_result_card(row) for row in results)
    split_options = _gallery_split_options(state, gallery_split)
    index_errors = "".join(
        f'<div class="warn">{html.escape(split)}: {html.escape(message)}</div>'
        for split, message in state.index_errors.items()
    )
    ready_status = f"{gallery_count:,} gallery images indexed on {state.device.type}"
    status_class = "alert" if error else "muted"
    status_text = error or ready_status
    preview = (
        f'<img class="preview-img" src="{query_preview}" alt="Query image preview">'
        if query_preview
        else '<div class="preview-empty">Choose an image to preview it instantly.</div>'
    )
    empty_results_class = "empty hidden" if results else "empty"
    empty_results_text = "Upload an image and run a query to see ranked gallery matches."
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ConText-CIRR | Retrieval Demo</title>
  <meta name="description" content="Interactive image-plus-text retrieval demo for the ConText-CIRR pipeline on the CIRR dataset.">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f7f5;
      --bg-accent: #e8f4ef;
      --panel: #ffffff;
      --panel-soft: #f8fbfa;
      --text: #18212f;
      --muted: #667085;
      --line: #d6e2db;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --accent-soft: #dff5ef;
      --danger: #b42318;
      --warn: #b54708;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      --radius: 16px;
      --radius-sm: 10px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 28%),
        linear-gradient(180deg, var(--bg-accent) 0%, var(--bg) 26%, #f8faf9 100%);
      color: var(--text);
      font-family: "Segoe UI", Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      border-bottom: 1px solid rgba(214, 226, 219, 0.9);
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(12px);
    }}
    .bar {{
      width: min(1200px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 20px 0 18px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
    }}
    .hero {{
      display: grid;
      gap: 6px;
      max-width: 680px;
    }}
    .eyebrow {{
      color: var(--accent-dark);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.05;
    }}
    .hero-copy {{
      color: #475467;
      font-size: 14px;
      line-height: 1.45;
    }}
    .muted, .alert {{
      font-size: 13px;
      font-weight: 700;
      border-radius: 999px;
      padding: 10px 14px;
      max-width: min(360px, 100%);
    }}
    .muted {{
      color: #0f513d;
      background: var(--accent-soft);
      border: 1px solid rgba(15, 118, 110, 0.18);
    }}
    .alert {{
      color: var(--danger);
      background: #fdecea;
      border: 1px solid rgba(180, 35, 24, 0.18);
    }}
    main {{
      width: min(1200px, calc(100vw - 32px));
      margin: 22px auto 36px;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 20px;
      align-items: start;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(214, 226, 219, 0.92);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .preview-shell {{
      padding: 14px 14px 0;
      display: grid;
      gap: 10px;
    }}
    .preview {{
      overflow: hidden;
      aspect-ratio: 1 / 1;
      border: 1px dashed rgba(15, 118, 110, 0.26);
      border-radius: 14px;
      background: linear-gradient(180deg, #edf7f3 0%, #f7fbfa 100%);
      display: grid;
      place-items: center;
    }}
    .preview-img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .preview-empty {{
      color: var(--muted);
      font-size: 13px;
      text-align: center;
      width: min(220px, 100%);
      line-height: 1.45;
    }}
    .preview-meta {{
      display: grid;
      gap: 4px;
      padding: 0 2px;
    }}
    .preview-title {{
      font-size: 13px;
      font-weight: 800;
    }}
    .preview-helper {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .query {{
      padding: 14px;
      display: grid;
      gap: 14px;
    }}
    label {{
      display: grid;
      gap: 6px;
      color: #344054;
      font-size: 13px;
      font-weight: 700;
    }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      padding: 11px 12px;
      color: var(--text);
      background: #fff;
      font: inherit;
      font-size: 14px;
      transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }}
    input:focus, select:focus, textarea:focus {{
      outline: none;
      border-color: rgba(15, 118, 110, 0.6);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
    }}
    textarea {{
      min-height: 110px;
      resize: vertical;
      line-height: 1.45;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 112px;
      gap: 10px;
    }}
    .search-actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .helper {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      max-width: 200px;
    }}
    button {{
      min-width: 126px;
      height: 42px;
      border: 0;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      color: #fff;
      font-size: 14px;
      font-weight: 800;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
      box-shadow: 0 12px 24px rgba(15, 118, 110, 0.22);
    }}
    button:hover {{ transform: translateY(-1px); }}
    button:disabled {{ opacity: 0.72; cursor: wait; transform: none; }}
    .warn-stack {{
      display: grid;
      gap: 8px;
    }}
    .warn {{
      color: var(--warn);
      font-size: 12px;
      font-weight: 700;
      padding: 8px 10px;
      border-radius: 10px;
      background: #fff7ed;
      border: 1px solid rgba(181, 71, 8, 0.16);
    }}
    .results-head {{
      padding: 18px 18px 14px;
      display: flex;
      justify-content: space-between;
      gap: 14px;
      border-bottom: 1px solid rgba(214, 226, 219, 0.9);
      align-items: flex-start;
    }}
    .results-copy {{
      display: grid;
      gap: 4px;
    }}
    .results-title {{
      font-size: 18px;
      font-weight: 800;
    }}
    .results-subtitle {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .grid {{
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(176px, 1fr));
      gap: 14px;
    }}
    .card {{
      border: 1px solid rgba(214, 226, 219, 0.95);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.04);
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: block;
      background: #eef2f6;
    }}
    .meta {{
      padding: 10px;
      display: grid;
      gap: 4px;
      min-width: 0;
    }}
    .name {{
      font-size: 12px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }}
    .score {{
      color: var(--muted);
      font-size: 12px;
    }}
    .empty {{
      padding: 56px 18px 20px;
      color: var(--muted);
      text-align: center;
      font-size: 14px;
      line-height: 1.45;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 900px) {{
      .bar {{
        align-items: flex-start;
        flex-direction: column;
      }}
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div class="hero">
        <div class="eyebrow">ConText-CIRR Workspace</div>
        <h1>Retrieval Demo</h1>
        <div class="hero-copy">Upload one reference image, keep it in the form, and iterate on modification text without choosing the file again.</div>
      </div>
      <div id="status-banner" class="{status_class}" data-ready="{html.escape(ready_status)}">{html.escape(status_text)}</div>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="preview-shell">
        <div id="preview-frame" class="preview">{preview}</div>
        <div class="preview-meta">
          <div class="preview-title">Reference preview</div>
          <div id="preview-helper" class="preview-helper">Preview appears immediately when you pick a file. The selected image stays in the form while you try new text queries.</div>
        </div>
      </div>
      <form id="query-form" class="query" action="/search" method="post" enctype="multipart/form-data">
        <label>Gallery split
          <select id="gallery-split" name="gallery_split">{split_options}</select>
        </label>
        <label>Reference image
          <input id="query-image" type="file" name="query_image" accept="image/*" required>
        </label>
        <label>Modification text
          <textarea id="query-text" name="text" required>{html.escape(text)}</textarea>
        </label>
        <div class="row">
          <label>Reference name
            <input id="reference-name" type="text" name="reference_name" value="{html.escape(reference_name)}">
          </label>
          <label>Top K
            <input id="top-k" type="number" name="top_k" min="1" max="50" value="{top_k}">
          </label>
        </div>
        <div class="search-actions">
          <button id="search-button" type="submit">Search</button>
          <div class="helper">Use Ctrl+Enter in the text box to search. If text already exists, choosing a new image will run a fresh search automatically.</div>
        </div>
        <div class="warn-stack">{index_errors}</div>
      </form>
    </section>
    <section class="panel">
      <div class="results-head">
        <div class="results-copy">
          <div class="results-title">Retrieved Gallery Images</div>
          <div class="results-subtitle">FAISS search over the selected CIRR gallery split.</div>
        </div>
        <div id="results-meta" class="muted">{html.escape(gallery_split)} &middot; {len(results)} shown</div>
      </div>
      <div id="results-empty" class="{empty_results_class}">{empty_results_text}</div>
      <div id="results-grid" class="grid">{result_cards}</div>
    </section>
  </main>
  <script>
    const form = document.querySelector("#query-form");
    const fileInput = document.querySelector("#query-image");
    const textInput = document.querySelector("#query-text");
    const referenceInput = document.querySelector("#reference-name");
    const galleryInput = document.querySelector("#gallery-split");
    const topKInput = document.querySelector("#top-k");
    const previewFrame = document.querySelector("#preview-frame");
    const previewHelper = document.querySelector("#preview-helper");
    const searchButton = document.querySelector("#search-button");
    const statusBanner = document.querySelector("#status-banner");
    const resultsGrid = document.querySelector("#results-grid");
    const resultsMeta = document.querySelector("#results-meta");
    const resultsEmpty = document.querySelector("#results-empty");
    let previewObjectUrl = null;
    let activeController = null;

    function escapeHtml(value) {{
      const replacements = {{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }};
      return String(value).replace(/[&<>"']/g, (char) => replacements[char]);
    }}

    function renderStatus(message, isError = false) {{
      statusBanner.className = isError ? "alert" : "muted";
      statusBanner.textContent = message || statusBanner.dataset.ready || "";
    }}

    function renderPreview(src, altText) {{
      if (!src) {{
        previewFrame.innerHTML = '<div class="preview-empty">Choose an image to preview it instantly.</div>';
        return;
      }}
      previewFrame.innerHTML = `<img class="preview-img" src="${{src}}" alt="${{escapeHtml(altText || 'Query image preview')}}">`;
    }}

    function updateReferenceFromFile(file) {{
      if (!file || referenceInput.value.trim()) {{
        return;
      }}
      referenceInput.value = file.name.replace(/\.[^.]+$/, "");
    }}

    function previewSelectedFile() {{
      const file = fileInput.files && fileInput.files[0];
      if (!file) {{
        return false;
      }}
      if (previewObjectUrl) {{
        URL.revokeObjectURL(previewObjectUrl);
      }}
      previewObjectUrl = URL.createObjectURL(file);
      renderPreview(previewObjectUrl, file.name);
      previewHelper.textContent = "Image is loaded locally and kept in the form while you retry text queries.";
      updateReferenceFromFile(file);
      return true;
    }}

    function renderResults(results, split) {{
      const rows = Array.isArray(results) ? results : [];
      resultsMeta.textContent = `${{split}} | ${{rows.length}} shown`;
      if (!rows.length) {{
        resultsGrid.innerHTML = "";
        resultsEmpty.classList.remove("hidden");
        return;
      }}
      resultsEmpty.classList.add("hidden");
      resultsGrid.innerHTML = rows.map((row) => {{
        const imageName = String(row.image_name || "");
        const gallerySplit = String(row.gallery_split || row.split || split || "val");
        const imageUrl = `/gallery/${{encodeURIComponent(gallerySplit)}}/${{encodeURIComponent(imageName)}}`;
        const score = Number(row.score || 0);
        const rank = Number(row.rank || 0);
        return `<article class="card"><img src="${{imageUrl}}" alt="${{escapeHtml(imageName)}}"><div class="meta"><div class="name">#${{rank}} ${{escapeHtml(imageName)}}</div><div class="score">score ${{score.toFixed(4)}}</div></div></article>`;
      }}).join("");
    }}

    async function submitSearch() {{
      const file = fileInput.files && fileInput.files[0];
      const textValue = textInput.value.trim();
      if (!file) {{
        renderStatus("Choose a reference image before searching.", true);
        return;
      }}
      if (!textValue) {{
        renderStatus("Enter modification text before searching.", true);
        return;
      }}
      previewSelectedFile();
      if (activeController) {{
        activeController.abort();
      }}
      const controller = new AbortController();
      activeController = controller;
      const previousLabel = searchButton.textContent;
      searchButton.disabled = true;
      searchButton.textContent = "Searching...";
      renderStatus("Searching the selected gallery...", false);
      try {{
        const formData = new FormData(form);
        const response = await fetch("/api/search", {{
          method: "POST",
          body: formData,
          signal: controller.signal,
        }});
        const payload = await response.json().catch(() => ({{ detail: "Search failed." }}));
        if (!response.ok) {{
          throw new Error(payload.detail || "Search failed.");
        }}
        referenceInput.value = payload.reference_name || referenceInput.value;
        renderResults(payload.results || [], payload.gallery_split || galleryInput.value);
        renderStatus(`${{(payload.results || []).length}} results loaded from ${{payload.gallery_split || galleryInput.value}}.`, false);
      }} catch (error) {{
        if (error.name === "AbortError") {{
          return;
        }}
        renderStatus(error.message || "Search failed.", true);
      }} finally {{
        if (activeController === controller) {{
          activeController = null;
        }}
        searchButton.disabled = false;
        searchButton.textContent = previousLabel;
      }}
    }}

    fileInput.addEventListener("change", () => {{
      const ready = previewSelectedFile();
      if (ready && textInput.value.trim()) {{
        submitSearch();
      }}
    }});

    form.addEventListener("submit", (event) => {{
      event.preventDefault();
      submitSearch();
    }});

    textInput.addEventListener("keydown", (event) => {{
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {{
        event.preventDefault();
        submitSearch();
      }}
    }});

    galleryInput.addEventListener("change", () => {{
      if (fileInput.files && fileInput.files.length && textInput.value.trim()) {{
        submitSearch();
      }}
    }});

    topKInput.addEventListener("change", () => {{
      if (fileInput.files && fileInput.files.length && textInput.value.trim()) {{
        submitSearch();
      }}
    }});
  </script>
</body>
</html>"""


def _gallery_split_options(state: DemoState, selected_split: str) -> str:
    rows = []
    for split, index_set in state.index_sets.items():
        selected = " selected" if split == selected_split else ""
        rows.append(
            f'<option value="{html.escape(split)}"{selected}>'
            f'{html.escape(split)} ({len(index_set.gallery_paths):,} images)</option>'
        )
    return "\n".join(rows)


def _result_card(row: dict[str, Any]) -> str:
    image_name = str(row["image_name"])
    gallery_split = str(row.get("gallery_split") or row.get("split") or "val")
    image_url = f"/gallery/{quote(gallery_split, safe='')}/{quote(image_name, safe='')}"
    score = float(row["score"])
    rank = int(row["rank"])
    return f"""<article class="card">
  <img src="{image_url}" alt="{html.escape(image_name)}">
  <div class="meta">
    <div class="name">#{rank} {html.escape(image_name)}</div>
    <div class="score">score {score:.4f}</div>
  </div>
</article>"""


def _search(
    state: DemoState,
    image_path: Path,
    text: str,
    top_k: int,
    gallery_split: str,
    reference_name: str | None,
) -> list[dict[str, Any]]:
    top_k = min(max(int(top_k), 1), 50)
    if gallery_split not in state.index_sets:
        raise KeyError(f"FAISS gallery is not loaded for split: {gallery_split}")
    index_set = state.index_sets[gallery_split]
    with state.lock:
        query_embedding = encode_query_image_text(state.model, image_path, text, state.device)
        results = search_loaded_index(
            index_set.faiss_index,
            index_set.mapping,
            query_embedding,
            top_k=top_k,
            reference_image_name=reference_name,
        )
    for row in results:
        row["gallery_split"] = gallery_split
    return results


def _load_index_sets(
    cfg: AppConfig,
    *,
    split: str,
    splits: list[str] | None,
    index_path: str | Path | None,
    mapping_path: str | Path | None,
) -> tuple[dict[str, IndexSet], dict[str, str]]:
    index_sets: dict[str, IndexSet] = {}
    index_errors: dict[str, str] = {}
    for requested_split in _requested_splits(split, splits, index_path):
        try:
            index_sets[requested_split] = _load_index_set(
                cfg,
                requested_split,
                index_path=index_path if requested_split == split else None,
                mapping_path=mapping_path if requested_split == split else None,
            )
        except Exception as exc:
            index_errors[requested_split] = str(exc)
    return index_sets, index_errors


def _create_demo_state(
    config_path: str | Path,
    checkpoint_path: str | Path,
    index_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    split: str = "val",
    splits: list[str] | None = None,
) -> DemoState:
    cfg = load_config(config_path)
    checkpoint = _resolve_project_path(checkpoint_path)
    _validate_file(checkpoint, "Checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_cir_module_from_checkpoint(checkpoint, cfg=cfg, map_location=device)
    model = module.model.to(device).eval()
    index_sets, index_errors = _load_index_sets(
        cfg,
        split=split,
        splits=splits,
        index_path=index_path,
        mapping_path=mapping_path,
    )
    if not index_sets:
        raise FileNotFoundError("No FAISS gallery indexes could be loaded.")

    upload_dir = _resolve_project_path(Path(cfg.project.output_dir) / "demo_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    return DemoState(
        cfg=cfg,
        checkpoint=checkpoint,
        upload_dir=upload_dir,
        device=device,
        model=model,
        index_sets=index_sets,
        index_errors=index_errors,
        lock=threading.Lock(),
    )


def _default_gallery_split(state: DemoState, split: str) -> str:
    return split if split in state.index_sets else next(iter(state.index_sets))


def _search_page_response(
    state: DemoState,
    query_image: UploadFile,
    text: str,
    top_k: int,
    gallery_split: str,
    reference_name: str,
) -> str:
    image_path = _save_upload(state.upload_dir, query_image)
    inferred_reference = reference_name.strip() or Path(query_image.filename or image_path.name).stem
    try:
        results = _search(state, image_path, text, top_k, gallery_split, inferred_reference)
        error = None
    except Exception as exc:
        results = []
        error = str(exc)
    return _page(
        state=state,
        results=results,
        query_preview=_image_data_url(image_path),
        text=text,
        reference_name=inferred_reference,
        gallery_split=gallery_split,
        top_k=min(max(int(top_k), 1), 50),
        error=error,
    )


def _search_api_response(
    state: DemoState,
    query_image: UploadFile,
    text: str,
    top_k: int,
    gallery_split: str,
    reference_name: str,
) -> JSONResponse:
    image_path = _save_upload(state.upload_dir, query_image)
    inferred_reference = reference_name.strip() or Path(query_image.filename or image_path.name).stem
    results = _search(state, image_path, text, top_k, gallery_split, inferred_reference)
    return JSONResponse({"results": results, "reference_name": inferred_reference, "gallery_split": gallery_split})


def _gallery_image_response(state: DemoState, gallery_split: str, image_name: str) -> FileResponse:
    index_set = state.index_sets.get(gallery_split)
    if index_set is None:
        raise HTTPException(status_code=404, detail=f"Gallery split not loaded: {gallery_split}")
    path = index_set.gallery_paths.get(image_name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")
    return FileResponse(path)


def _health_payload(state: DemoState) -> dict[str, Any]:
    return {
        "status": "ok",
        "device": state.device.type,
        "indexes": {
            split_name: {
                "gallery_images": len(index_set.gallery_paths),
                "index": str(index_set.index_path),
                "mapping": str(index_set.mapping_path),
            }
            for split_name, index_set in state.index_sets.items()
        },
        "index_errors": state.index_errors,
    }


def create_app(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    index_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    split: str = "val",
    splits: list[str] | None = None,
) -> FastAPI:
    state = _create_demo_state(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        index_path=index_path,
        mapping_path=mapping_path,
        split=split,
        splits=splits,
    )
    app = FastAPI(title="CIR Retrieval Demo")
    app.state.demo = state

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> str:
        default_split = _default_gallery_split(state, split)
        return _page(state=state, gallery_split=default_split)

    @app.post("/search", response_class=HTMLResponse)
    def search_page(
        query_image: Annotated[UploadFile, File(...)],
        text: Annotated[str, Form(...)],
        top_k: Annotated[int, Form()] = 10,
        gallery_split: Annotated[str, Form()] = "val",
        reference_name: Annotated[str, Form()] = "",
    ) -> str:
        return _search_page_response(state, query_image, text, top_k, gallery_split, reference_name)

    @app.post("/api/search", responses={400: {"description": "Bad request"}})
    def search_api(
        query_image: Annotated[UploadFile, File(...)],
        text: Annotated[str, Form(...)],
        top_k: Annotated[int, Form()] = 10,
        gallery_split: Annotated[str, Form()] = "val",
        reference_name: Annotated[str, Form()] = "",
    ) -> JSONResponse:
        try:
            return _search_api_response(state, query_image, text, top_k, gallery_split, reference_name)
        except Exception as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get(
        "/gallery/{gallery_split}/{image_name}",
        responses={404: {"description": "Not found"}},
    )
    def gallery_image(gallery_split: str, image_name: str) -> FileResponse:
        return _gallery_image_response(state, gallery_split, image_name)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return _health_payload(state)

    return app


def main() -> None:
    args = parse_args()
    app = create_app(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        index_path=args.index,
        mapping_path=args.mapping,
        split=args.split,
        splits=args.splits,
    )
    print(f"Demo URL: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
