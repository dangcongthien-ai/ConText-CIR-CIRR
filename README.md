# CIR_Pipeline: ConText-CIR-like CIRR Retrieval

Professional PyTorch Lightning pipeline for Composed Image Retrieval on CIRR.
The model follows the ConText-CIR idea: encode a query image and modification
text, fuse visual and text tokens with cross-attention, pool into a shared
retrieval embedding, and train with contrastive retrieval plus optional Text
Concept-Consistency loss.

## Architecture

Training:

1. Query branch computes `r_q = f(I_q, T)` from CLIP visual tokens, CLIP text
   tokens, cross-attention, and attention pooling.
2. Target branch computes `r_t = f(I_t, "")`, using the exact same fusion
   network and blank text.
3. Query-negative branch computes `r_q_neg = f(I_q, "")`.
4. Loss is `L_total = L_cont + lambda_cc * L_CC`, where `L_cont` is InfoNCE
   over in-batch targets plus query-image negatives, and `L_CC` compares
   full-caption and isolated noun-phrase attention maps.

Inference:

1. Encode every official gallery image as `f(C_i, "")`.
2. L2-normalize embeddings and store them in a FAISS `IndexFlatIP` by default.
3. Encode the query as `f(I_q, T)`, search FAISS, and remove the reference
   image from returned results when it appears in the gallery.

## Dataset Layout

Expected local layout:

```text
CIRR/
  train/
  dev/
  test1/
  cirr/
    image_splits/split.rc2.train.json
    image_splits/split.rc2.val.json
    image_splits/split.rc2.test1.json
    captions/cap.rc2.train.json
    captions/cap.rc2.val.json
    captions/cap.rc2.test1.json
    captions_ext/cap.ext.rc2.train.json
    captions_ext/cap.ext.rc2.val.json
    captions_ext/cap.ext.rc2.test1.json
```

The pipeline never trains, validates, indexes, or retrieves by scanning image
folders. It resolves image files only through `image_splits/split.rc2.*.json`.
This matters because the local image folders may be supersets with extra PNGs
that do not belong to CIRR rc2 annotations.

Local metadata facts baked into inspection expectations:

| Split | Official images | Caption pairs | Notes |
|---|---:|---:|---|
| train | 16,939 | 28,225 | train images are under subfolders |
| val | 2,297 | 4,181 | metadata `val` maps to local `dev` |
| test1 | 2,315 | 4,148 | one empty caption is handled safely |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you want GPU FAISS, install a `faiss-gpu` build matching your CUDA setup
instead of `faiss-cpu`.

## COMET

No API key is hardcoded. The easiest setup is to copy the example env file:

```bash
copy .env.example .env
```

Then edit `.env`:

```bash
COMET_API_KEY=your_key
COMET_WORKSPACE=your_workspace
COMET_PROJECT_NAME=ConText-CIR-CIRR
```

The project auto-loads `.env` from the repository root before creating the
COMET logger. You can also set environment variables manually:

```bash
set COMET_API_KEY=your_key
set COMET_WORKSPACE=your_workspace
set COMET_PROJECT_NAME=ConText-CIR-CIRR
```

If `logger.name=comet` and `COMET_API_KEY` is missing, training prints a warning
and continues without crashing. You can also set `logger.name: "tensorboard"` or
`logger.name: "none"` in the YAML.

## Commands

Inspect dataset:

```bash
python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir.yaml
```

Train smoke run:

```bash
python scripts/train.py --config configs/cirr_context_cir.yaml --fast_dev_run
```

Train full run:

```bash
python scripts/train.py --config configs/cirr_context_cir.yaml
```

Validate checkpoint:

```bash
python scripts/validate.py --config configs/cirr_context_cir.yaml --checkpoint checkpoints/context_cir_cirr_baseline/best.ckpt
```

Build FAISS index:

```bash
python scripts/build_faiss_index.py --config configs/cirr_context_cir.yaml --checkpoint checkpoints/context_cir_cirr_baseline/best.ckpt --split val
```

Retrieve:

```bash
python scripts/retrieve.py --config configs/cirr_context_cir.yaml --checkpoint checkpoints/context_cir_cirr_baseline/best.ckpt --index outputs/faiss/cirr_val/index.faiss --mapping outputs/faiss/cirr_val/mapping.json --query_image sample.png --text "make the scene brighter" --top_k 10
```

Create CIRR test1 submission:

```bash
python scripts/make_cirr_submission.py --config configs/cirr_context_cir.yaml --checkpoint checkpoints/context_cir_cirr_baseline/best.ckpt
```

## File And Folder Purpose

| Path | Purpose |
|---|---|
| `configs/cirr_context_cir.yaml` | Full train/eval/retrieval/logger/path config. |
| `src/cir_pipeline/config.py` | Load and validate config with dataclasses. |
| `src/cir_pipeline/seed.py` | Seed Python, NumPy, and Torch. |
| `src/cir_pipeline/data/cirr_dataset.py` | CIRR Dataset reading only official image_splits plus captions. |
| `src/cir_pipeline/data/cirr_datamodule.py` | Lightning DataModule for train/val/test dataloaders. |
| `src/cir_pipeline/data/transforms.py` | CLIP resize/crop/normalize image transforms. |
| `src/cir_pipeline/data/collate.py` | Collate image tensors, captions, noun phrases, pair ids, names, and img_set. |
| `src/cir_pipeline/data/validate_cirr_files.py` | Read-only metadata/image availability inspection. |
| `src/cir_pipeline/models/encoders.py` | HuggingFace CLIP image and text token encoders. |
| `src/cir_pipeline/models/attention.py` | Multi-head cross-attention and learned attention pooling. |
| `src/cir_pipeline/models/context_cir.py` | Main `ConTextCIR` model with `forward_query`, `encode_gallery`, and attention output. |
| `src/cir_pipeline/models/losses.py` | InfoNCE retrieval loss and Text Concept-Consistency loss. |
| `src/cir_pipeline/text/noun_phrase_parser.py` | Noun phrase extraction via Stanza, NLTK, or regex fallback. |
| `src/cir_pipeline/text/token_alignment.py` | Safe noun phrase to CLIP token index mapping via tokenizer offsets. |
| `src/cir_pipeline/engine/train_module.py` | LightningModule for train/validation step, optimizer, and scheduler. |
| `src/cir_pipeline/engine/metrics.py` | Recall@K, RecallSubset@K, and similarity utilities. |
| `src/cir_pipeline/engine/evaluator.py` | Full-gallery validation retrieval evaluator. |
| `src/cir_pipeline/engine/comet_logger.py` | COMET/TensorBoard logger creation and artifact helpers. |
| `src/cir_pipeline/retrieval/faiss_index.py` | FAISS create/save/load/search wrapper. |
| `src/cir_pipeline/retrieval/build_gallery.py` | Encode gallery images with `f(C_i, "")` and persist FAISS artifacts. |
| `src/cir_pipeline/retrieval/search.py` | Encode query `f(I_q,T)`, search FAISS, and remove reference image. |
| `src/cir_pipeline/utils/io.py` | JSON and filesystem helpers. |
| `src/cir_pipeline/utils/device.py` | Recursive batch-to-device helper. |
| `src/cir_pipeline/utils/logging.py` | Logging setup helpers. |
| `scripts/train.py` | CLI training entrypoint. |
| `scripts/validate.py` | CLI checkpoint validation. |
| `scripts/build_faiss_index.py` | CLI FAISS index builder. |
| `scripts/retrieve.py` | CLI image plus text retrieval demo. |
| `scripts/make_cirr_submission.py` | Create CIRR test1 recall and recall_subset JSON submissions. |
| `scripts/inspect_cirr_dataset.py` | CLI dataset inspection. |
| `tests/` | Unit tests for dataset, losses, and FAISS. |
| `data/README.md` | Dataset placement notes; no raw data is stored here. |
| `outputs/.gitkeep` | Keeps generated output folder in the project tree. |
| `checkpoints/.gitkeep` | Keeps checkpoint folder in the project tree. |

## Troubleshooting

`COMET_API_KEY` missing:
Set COMET env vars, use `logger.offline: true`, or switch to `logger.name:
"none"`.

Stanza model missing:
The parser warns and falls back to NLTK or regex. For better noun phrases, run
`python -c "import stanza; stanza.download('en')"`.

CUDA OOM:
Lower `train.batch_size`, `train.val_batch_size`, or use gradient accumulation
by adding it to the Trainer in `scripts/train.py`.

FAISS install issue:
Use `pip install faiss-cpu` on CPU machines. For CUDA, install a compatible
`faiss-gpu` package manually.

Missing image path:
Run `python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir.yaml`.
The loader expects every image path declared in `image_splits` to exist under
`data.root`.
