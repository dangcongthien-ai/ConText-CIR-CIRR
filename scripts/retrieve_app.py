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
    status = (
        f'<div class="alert">{html.escape(error)}</div>'
        if error
        else f'<div class="muted">{gallery_count:,} gallery images indexed on {state.device.type}</div>'
    )
    preview = (
        f'<img class="preview-img" src="{query_preview}" alt="Query image preview">'
        if query_preview
        else '<div class="preview-empty">No image</div>'
    )
    empty_results = (
        '<div class="empty">No results yet</div>'
        if not results and not error
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CIR Retrieval Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-dark: #115e59;
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
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    .bar {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 14px 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 18px auto 32px;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .query {{
      padding: 14px;
      display: grid;
      gap: 12px;
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
      border-radius: 6px;
      padding: 10px;
      color: var(--text);
      background: #fff;
      font: inherit;
      font-size: 14px;
    }}
    textarea {{
      min-height: 92px;
      resize: vertical;
      line-height: 1.35;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 96px;
      gap: 10px;
    }}
    button {{
      height: 40px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    .preview {{
      overflow: hidden;
      aspect-ratio: 1 / 1;
      border-bottom: 1px solid var(--line);
      background: #eef2f6;
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
    }}
    .alert {{
      color: var(--danger);
      font-size: 13px;
      font-weight: 700;
    }}
    .warn {{
      color: var(--warn);
      font-size: 12px;
      font-weight: 700;
    }}
    .results-head {{
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
    }}
    .results-title {{
      font-size: 14px;
      font-weight: 700;
    }}
    .grid {{
      padding: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
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
      background: #eef2f6;
    }}
    .meta {{
      padding: 9px;
      display: grid;
      gap: 4px;
      min-width: 0;
    }}
    .name {{
      font-size: 12px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .score {{
      color: var(--muted);
      font-size: 12px;
    }}
    .empty {{
      padding: 48px 14px;
      color: var(--muted);
      text-align: center;
      font-size: 14px;
    }}
    @media (max-width: 820px) {{
      .bar {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .muted {{ white-space: normal; }}
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <h1>CIR Retrieval Demo</h1>
      {status}
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="preview">{preview}</div>
      <form class="query" action="/search" method="post" enctype="multipart/form-data">
        <label>Gallery
          <select name="gallery_split">{split_options}</select>
        </label>
        <label>Reference image
          <input type="file" name="query_image" accept="image/*" required>
        </label>
        <label>Modification text
          <textarea name="text" required>{html.escape(text)}</textarea>
        </label>
        <div class="row">
          <label>Reference name
            <input type="text" name="reference_name" value="{html.escape(reference_name)}">
          </label>
          <label>Top K
            <input type="number" name="top_k" min="1" max="50" value="{top_k}">
          </label>
        </div>
        <button type="submit">Search</button>
        {index_errors}
      </form>
    </section>
    <section class="panel">
      <div class="results-head">
        <div class="results-title">Results</div>
        <div class="muted">{html.escape(gallery_split)} &middot; {len(results)} shown</div>
      </div>
      {empty_results}
      <div class="grid">{result_cards}</div>
    </section>
  </main>
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

    @app.post("/api/search")
    def search_api(
        query_image: Annotated[UploadFile, File(...)],
        text: Annotated[str, Form(...)],
        top_k: Annotated[int, Form()] = 10,
        gallery_split: Annotated[str, Form()] = "val",
        reference_name: Annotated[str, Form()] = "",
    ) -> JSONResponse:
        return _search_api_response(state, query_image, text, top_k, gallery_split, reference_name)

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
