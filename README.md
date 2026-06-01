# Context-CIR-CIRR

Pipeline Composed Image Retrieval cho CIRR với backbone `openai/clip-vit-base-patch32`.

Hiện chỉ giữ 2 hướng chạy cần so sánh:

| Variant | Config | Loss | Tên run/checkpoint/FAISS |
|---|---|---|---|
| CL-only | `configs/cirr_context_cir_cl.yaml` | Contrastive retrieval | `cirr_vitb32_cl` |
| CL+CC | `configs/cirr_context_cir_cl_cc.yaml` | Contrastive retrieval + Text Concept-Consistency | `cirr_vitb32_cl_cc` |

Không chạy `CC-only` nữa, vì Text-CC một mình không học retrieval target. Text-CC chỉ nên xem là regularizer để so `CL-only` với `CL+CC`.

## Port Map

Để tránh đụng port khi mở nhiều UI cùng lúc, dùng cố định:

| UI | Port | Ghi chú |
|---|---|---|
| Retrieval demo `CL-only` | `8000` | App retrieval đầu tiên |
| Heatmap UI | `8001` | So `CL-only` và `CL+CC` |
| Dataset viewer | `8002` | Xem pair, caption, target |
| Retrieval demo `CL+CC` | `8003` | Dùng khi cần mở song song với `CL-only` |

Nếu chỉ chạy một app, có thể bỏ `--port` để dùng default của script. Nếu chạy song song nhiều app, nên bám đúng bảng trên.

## 1. Chuẩn Bị

```powershell
cd D:\1_Documents\Nam_3_HK2\Advanced_Computer_Vision\vscode\CIR_Pipeline
conda activate AI
pip install -r requirements.txt
```

Nếu dùng COMET, tạo `.env`:

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

Nếu không dùng COMET, sửa config:

```yaml
logger:
  name: "none"
```

## 2. Kiểm Tra Dataset

Dataset cần có cấu trúc:

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

Mở giao diện xem reference image, caption và target image:

```powershell
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml
```

Mở:

```text
http://127.0.0.1:8002
```

Nếu muốn list nhiều sample hơn:

```powershell
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml --limit 1000
```

## 3. Test Nhanh

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml --fast_dev_run
pytest -q
```

## 4. Train 2 Model

### Train CL-only

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

### Train CL+CC

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

## 5. Validate Trên Val

Validate CL-only:

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
```

Validate CL+CC:

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
```

Metric chính để so:

```text
recall@1
recall@5
recall@10
recall@50
recall_subset@1
recall_subset@2
recall_subset@3
```

Kết luận nên dựa vào:

```text
CL-only vs CL+CC
```

Nếu `CL+CC` không hơn `CL-only`, nghĩa là Text-CC hiện tại chưa giúp retrieval.

## 6. Build FAISS Cho Val Và Test1

FAISS lưu riêng theo variant:

```text
outputs/faiss/cirr_vitb32_cl/cirr_val/
outputs/faiss/cirr_vitb32_cl/cirr_test1/
outputs/faiss/cirr_vitb32_cl_cc/cirr_val/
outputs/faiss/cirr_vitb32_cl_cc/cirr_test1/
```

Build FAISS CL-only:

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split test1
```

Build FAISS CL+CC:

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split test1
```

Mỗi thư mục FAISS có:

```text
index.faiss
mapping.json
metadata.json
```

COMET FAISS logs sẽ có dạng:

```text
cirr_vitb32_cl_faiss_val
cirr_vitb32_cl_faiss_test1
cirr_vitb32_cl_cc_faiss_val
cirr_vitb32_cl_cc_faiss_test1
```

## 7. Demo Retrieval

Demo cho phép chọn gallery `val` hoặc `test1`.

Chạy demo CL-only:

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --splits val test1 --port 8000
```

Chạy demo CL+CC:

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --splits val test1 --port 8003
```

Mở lần lượt:

```text
CL-only: http://127.0.0.1:8000
CL+CC:  http://127.0.0.1:8003
```

Nếu chỉ muốn truyền một FAISS index cụ thể:

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val --index outputs/faiss/cirr_vitb32_cl_cc/cirr_val/index.faiss --mapping outputs/faiss/cirr_vitb32_cl_cc/cirr_val/mapping.json --port 8003
```

## 8. Heatmap UI

Heatmap UI hiện chỉ so 2 model:

```text
CL-only
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
checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt
```

Nếu checkpoint nằm chỗ khác:

```powershell
python scripts/generate_loss_heatmaps.py --config configs/cirr_context_cir_cl_cc.yaml --split val --cl_checkpoint path\to\cl.ckpt --cl_cc_checkpoint path\to\cl_cc.ckpt
```

Output:

```text
outputs/heatmaps/cl/
outputs/heatmaps/cl_cc/
outputs/heatmaps/reports/
```

## 9. Export Noun Phrases

`--backend both` nghĩa là export cả `stanza` và `regex`, rồi tạo thêm file so sánh.

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

Nếu chỉ muốn Stanza:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split all --backend stanza --per_backend_dirs
```

Nếu chỉ muốn regex:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split all --backend regex --per_backend_dirs
```

Chuẩn bị Stanza resources:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --backend stanza --prepare_stanza
```

## 10. Tạo Submission Test1

Submission CL-only:

```powershell
python scripts/make_cirr_submission.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --submission_name cirr_vitb32_cl
```

Submission CL+CC:

```powershell
python scripts/make_cirr_submission.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --submission_name cirr_vitb32_cl_cc
```

Output:

```text
outputs/submissions/cirr/<submission_name>.json
outputs/submissions/cirr/subset_<submission_name>.json
```

## 11. Lệnh Chạy Nhanh Từ Đầu

```powershell
python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir_cl_cc.yaml

python scripts/train.py --config configs/cirr_context_cir_cl.yaml
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml

python scripts/validate.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val

python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split test1
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split test1

python scripts/retrieve_app.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --splits val test1 --port 8000
python scripts/generate_loss_heatmaps.py --config configs/cirr_context_cir_cl_cc.yaml --split val --port 8001
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml --port 8002
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --splits val test1 --port 8003
```

## 12. File Quan Trọng

| Path | Chức năng |
|---|---|
| `configs/cirr_context_cir_cl.yaml` | Config CL-only |
| `configs/cirr_context_cir_cl_cc.yaml` | Config CL+CC |
| `scripts/train.py` | Train model |
| `scripts/validate.py` | Validate checkpoint |
| `scripts/build_faiss_index.py` | Build FAISS gallery index |
| `scripts/retrieve_app.py` | Demo retrieval FastAPI |
| `scripts/dataset_app.py` | Visualize dataset |
| `scripts/generate_loss_heatmaps.py` | Visualize heatmap |
| `scripts/export_noun_phrases.py` | Export Stanza/regex noun phrases |
| `scripts/make_cirr_submission.py` | Tạo submission JSON |
| `src/cir_pipeline/engine/train_module.py` | LightningModule và loss tổng |
| `src/cir_pipeline/models/losses.py` | Contrastive loss và Text-CC loss |

## 13. Lỗi Thường Gặp

### COMET không có API key

Sửa config:

```yaml
logger:
  name: "none"
```

Hoặc kiểm tra `.env`.

### Chưa build FAISS nhưng mở retrieval demo

Build FAISS trước:

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_vitb32_cl_cc/cirr_vitb32_cl_cc-best.ckpt --split test1
```

### Stanza lỗi trên Windows

Chuẩn bị resource:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --backend stanza --prepare_stanza
```

Nếu vẫn lỗi:

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
