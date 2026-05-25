# CIR_Pipeline

Pipeline Composed Image Retrieval cho CIRR, dùng backbone `openai/clip-vit-base-patch32`.
Dự án hiện có 3 biến thể train chính:

| Variant | Config | Loss | COMET/checkpoint/FAISS name |
|---|---|---|---|
| CL | `configs/cirr_context_cir_cl.yaml` | Contrastive retrieval only | `cirr_vitb32_cl` |
| CC | `configs/cirr_context_cir_cc.yaml` | Text Concept-Consistency only | `cirr_vitb32_cc` |
| CL+CC | `configs/cirr_context_cir_cl_cc.yaml` | Contrastive + Text Concept-Consistency | `cirr_vitb32_cl_cc` |

Các config cũ đã được bỏ để tránh nhầm hướng chạy.

## 1. Chuẩn Bị Môi Trường

Từ thư mục project:

```powershell
cd D:\1_Documents\Nam_3_HK2\Advanced_Computer_Vision\vscode\CIR_Pipeline
conda activate AI
pip install -r requirements.txt
```

Nếu dùng COMET, tạo hoặc sửa file `.env` ở thư mục gốc:

```powershell
copy .env.example .env
notepad .env
```

Nội dung `.env`:

```text
COMET_API_KEY=your_key
COMET_WORKSPACE=your_workspace
COMET_PROJECT_NAME=ConText-CIR-CIRR
```

Nếu không muốn log COMET, sửa trong YAML:

```yaml
logger:
  name: "none"
```

## 2. Kiểm Tra Dataset

Cấu trúc cần có:

```text
CIRR/
  train/
  dev/
  test1/
  cirr/
    image_splits/
    captions/
    captions_ext/
```

Kiểm tra metadata và ảnh:

```powershell
python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir_cl_cc.yaml
```

Mở giao diện xem dataset, gồm ảnh query/reference, caption và ảnh target:

```powershell
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml
```

Sau đó mở:

```text
http://127.0.0.1:8002
```

## 3. Test Nhanh Trước Khi Train

Chạy smoke test 1 batch để kiểm tra pipeline không lỗi:

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml --fast_dev_run
```

Chạy unit tests:

```powershell
pytest -q
```

## 4. Train 3 Model

### 4.1. Train CL

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl.yaml
```

Checkpoint tốt nhất:

```text
checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt
```

COMET run:

```text
cirr_vitb32_cl
```

### 4.2. Train CC

```powershell
python scripts/train.py --config configs/cirr_context_cir_cc.yaml
```

Checkpoint tốt nhất:

```text
checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt
```

COMET run:

```text
cirr_vitb32_cc
```

Lưu ý: CC-only là ablation để kiểm tra tác dụng Text-CC. Nó không trực tiếp tối ưu retrieval target như CL, nên kết quả retrieval có thể yếu.

### 4.3. Train CL+CC

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml
```

Checkpoint tốt nhất:

```text
checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt
```

COMET run:

```text
cirr_vitb32_cl_cc
```

## 5. Validate 3 Model Trên Val

### 5.1. Validate CL

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
```

### 5.2. Validate CC

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cc.yaml --checkpoint checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt --split val
```

### 5.3. Validate CL+CC

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
```

Metrics chính cần so sánh:

```text
recall@1
recall@5
recall@10
recall@50
recall_subset@1
recall_subset@2
recall_subset@3
```

## 6. Build FAISS Cho Val Và Test1

FAISS được lưu riêng theo variant:

```text
outputs/faiss/cirr_vitb32_cl/cirr_val/
outputs/faiss/cirr_vitb32_cl/cirr_test1/
outputs/faiss/cirr_vitb32_cc/cirr_val/
outputs/faiss/cirr_vitb32_cc/cirr_test1/
outputs/faiss/cirr_vitb32_cl_cc/cirr_val/
outputs/faiss/cirr_vitb32_cl_cc/cirr_test1/
```

### 6.1. Build FAISS CL

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split test1
```

COMET FAISS logs:

```text
cirr_vitb32_cl_faiss_val
cirr_vitb32_cl_faiss_test1
```

### 6.2. Build FAISS CC

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cc.yaml --checkpoint checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cc.yaml --checkpoint checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt --split test1
```

COMET FAISS logs:

```text
cirr_vitb32_cc_faiss_val
cirr_vitb32_cc_faiss_test1
```

### 6.3. Build FAISS CL+CC

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split test1
```

COMET FAISS logs:

```text
cirr_vitb32_cl_cc_faiss_val
cirr_vitb32_cl_cc_faiss_test1
```

Mỗi thư mục FAISS có:

```text
index.faiss
mapping.json
metadata.json
```

## 7. Mở Demo Retrieval

Demo cho phép upload/chọn ảnh query, nhập caption thay đổi, chọn gallery `val` hoặc `test1`, rồi trả ảnh retrieval từ FAISS.

### 7.1. Demo CL

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --splits val test1 --port 8000
```

### 7.2. Demo CC

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cc.yaml --checkpoint checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt --splits val test1 --port 8000
```

### 7.3. Demo CL+CC

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --splits val test1 --port 8000
```

Mở:

```text
http://127.0.0.1:8000
```

Nếu chỉ muốn dùng một FAISS index cụ thể:

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val --index outputs/faiss/cirr_vitb32_cl_cc/cirr_val/index.faiss --mapping outputs/faiss/cirr_vitb32_cl_cc/cirr_val/mapping.json
```

## 8. Mở Heatmap UI

Heatmap UI show sample bằng thumbnail ảnh, rồi sinh heatmap cho:

```text
CL
CC
CL+CC
```

Chạy:

```powershell
python scripts/generate_loss_heatmaps.py --config configs/cirr_context_cir_cl_cc.yaml --split val --port 8001
```

Mở:

```text
http://127.0.0.1:8001
```

Mặc định UI tìm checkpoint:

```text
checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt
checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt
checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt
```

Nếu checkpoint nằm chỗ khác:

```powershell
python scripts/generate_loss_heatmaps.py --config configs/cirr_context_cir_cl_cc.yaml --split val --cl_checkpoint path\to\cl.ckpt --cc_checkpoint path\to\cc.ckpt --cl_cc_checkpoint path\to\cl_cc.ckpt
```

Output heatmap:

```text
outputs/heatmaps/cl/
outputs/heatmaps/cc/
outputs/heatmaps/cl_cc/
outputs/heatmaps/reports/
```

## 9. Export Stanza Và Regex Cho Cả Dataset

Xuất noun phrase bằng cả Stanza và regex cho train/val/test1:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split all --backend both --per_backend_dirs
```

Output chính:

```text
outputs/noun_phrases/cirr_train_stanza.json
outputs/noun_phrases/cirr_train_regex.json
outputs/noun_phrases/cirr_train_stanza_regex.json
outputs/noun_phrases/cirr_val_stanza.json
outputs/noun_phrases/cirr_val_regex.json
outputs/noun_phrases/cirr_val_stanza_regex.json
outputs/noun_phrases/cirr_test1_stanza.json
outputs/noun_phrases/cirr_test1_regex.json
outputs/noun_phrases/cirr_test1_stanza_regex.json
```

Compatibility outputs:

```text
outputs/stanza/stanza_output_train.json
outputs/stanza/stanza_output_val.json
outputs/stanza/stanza_output_test1.json
outputs/regex/regex_output_train.json
outputs/regex/regex_output_val.json
outputs/regex/regex_output_test1.json
```

Nếu Stanza chưa có resource:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --backend stanza --prepare_stanza
```

## 10. Tạo Submission CIRR Test1

Thường dùng model tốt nhất theo validation, ví dụ CL+CC:

```powershell
python scripts/make_cirr_submission.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --submission_name cirr_vitb32_cl_cc
```

Output:

```text
outputs/submissions/cirr/cirr_vitb32_cl_cc.json
outputs/submissions/cirr/subset_cirr_vitb32_cl_cc.json
```

Tạo submission cho CL:

```powershell
python scripts/make_cirr_submission.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --submission_name cirr_vitb32_cl
```

Tạo submission cho CC:

```powershell
python scripts/make_cirr_submission.py --config configs/cirr_context_cir_cc.yaml --checkpoint checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt --submission_name cirr_vitb32_cc
```

## 11. Lệnh Chạy Nhanh Theo Thứ Tự

Nếu muốn chạy trọn bộ từ đầu:

```powershell
python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir_cl_cc.yaml

python scripts/train.py --config configs/cirr_context_cir_cl.yaml
python scripts/train.py --config configs/cirr_context_cir_cc.yaml
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml

python scripts/validate.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/validate.py --config configs/cirr_context_cir_cc.yaml --checkpoint checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt --split val
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val

python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split test1
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cc.yaml --checkpoint checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cc.yaml --checkpoint checkpoints/cirr_vitb32_cc/cirr_vitb32_cc-best.ckpt --split test1
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split test1

python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml --port 8002
python scripts/generate_loss_heatmaps.py --config configs/cirr_context_cir_cl_cc.yaml --split val --port 8001
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --splits val test1 --port 8000
```

## 12. File Quan Trọng

| Path | Chức năng |
|---|---|
| `configs/cirr_context_cir_cl.yaml` | Config CL-only |
| `configs/cirr_context_cir_cc.yaml` | Config CC-only |
| `configs/cirr_context_cir_cl_cc.yaml` | Config CL+CC |
| `scripts/train.py` | Train model |
| `scripts/validate.py` | Validate checkpoint trên val |
| `scripts/build_faiss_index.py` | Build FAISS gallery index |
| `scripts/retrieve_app.py` | Demo retrieval FastAPI |
| `scripts/dataset_app.py` | Visualize dataset |
| `scripts/generate_loss_heatmaps.py` | Visualize heatmap |
| `scripts/export_noun_phrases.py` | Export Stanza/regex noun phrases |
| `scripts/make_cirr_submission.py` | Tạo submission JSON |
| `src/cir_pipeline/engine/train_module.py` | LightningModule và loss tổng |
| `src/cir_pipeline/models/losses.py` | Contrastive loss và Text-CC loss |
| `src/cir_pipeline/data/cirr_dataset.py` | Reader CIRR từ metadata chính thức |

## 13. Lỗi Thường Gặp

### COMET không có API key

Nếu không cần COMET:

```yaml
logger:
  name: "none"
```

Nếu cần COMET, kiểm tra `.env`:

```text
COMET_API_KEY=...
COMET_WORKSPACE=...
COMET_PROJECT_NAME=ConText-CIR-CIRR
```

### Chưa build FAISS nhưng mở retrieval demo

Build đủ FAISS trước:

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split test1
```

### Stanza lỗi trên Windows

Chuẩn bị resource:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --backend stanza --prepare_stanza
```

Nếu vẫn lỗi, dùng regex hoặc `--allow_fallback` khi export:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split all --backend both --allow_fallback
```

### CUDA out of memory

Giảm trong YAML:

```yaml
train:
  batch_size: 32
  val_batch_size: 64
  accumulate_grad_batches: 2
```

Hoặc giảm số layer trainable:

```yaml
model:
  trainable_vision_layers: 1
  trainable_text_layers: 1
```

### Checkpoint cũ không load được

Dùng checkpoint đúng với config variant. Ví dụ CL+CC:

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
```
