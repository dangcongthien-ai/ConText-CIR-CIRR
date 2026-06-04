# ConText-CIR-CIRR

Pipeline **Composed Image Retrieval** cho CIRR sử dụng `openai/clip-vit-large-patch14` + `targetpad`.

Repo này hiện giữ một so sánh đơn giản:

| Biến thể | Config                                | Mục đích                           | Tên run                  |
| -------- | ------------------------------------- | ---------------------------------- | ------------------------ |
| CL-only  | `configs/cirr_context_cir_cl.yaml`    | Baseline retrieval                 | `cirr_vitb32_cl`         |
| CL+CC    | `configs/cirr_context_cir_cl_cc.yaml` | Retrieval + Text-CC regularization | `cirr_clip_l14_targetpad_cl_cc_t2i` |

Config `CL+CC` hiện tại là hướng parser duy nhất muốn kiểm tra so với `CL-only`:

* `noun_phrase_backend=stanza`
* `noun_phrase_extractor=constituency_minimal_np`
* `image_transform=targetpad`
* `backbone=openai/clip-vit-large-patch14`
* `lambda_cc=0.08`
* `text_cc_clean_only=true`
* `text_cc_clean_min_confidence=0.7`
* `text_cc_attention_direction=text_to_image`

## 1. Cài đặt

```powershell
cd D:\1_Documents\Nam_3_HK2\Advanced_Computer_Vision\vscode\ConText-CIR-CIRR
conda activate AI
pip install -r requirements.txt
```

Cài đặt Comet tùy chọn:

```powershell
copy .env.example .env
notepad .env
```

Ví dụ file `.env`:

```text
COMET_API_KEY=your_key
COMET_WORKSPACE=your_workspace
COMET_PROJECT_NAME=ConText-CIR-CIRR
```

Nếu không muốn dùng Comet logging, đặt giá trị sau trong config bạn dùng:

```yaml
logger:
  name: "none"
```

## 2. Cấu trúc dataset

Cấu trúc thư mục mong đợi:

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

Kiểm tra nhanh dataset:

```powershell
python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir_cl_cc.yaml
```

Chạy trình xem dataset:

```powershell
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml --port 8002
```

Mở:

```text
Local: http://127.0.0.1:8002
LAN:   http://<your-lan-ip>:8002
```

## 3. Quy trình chạy từ đầu đến cuối

### Bước 1: Export noun phrases

Chạy bước này trước khi train `CL+CC`.

Chuẩn bị tài nguyên Stanza một lần nếu cần:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --backend stanza --prepare_stanza
```

Export Stanza noun phrases cho `val`:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split val --backend stanza --per_backend_dirs
```

Output chính:

```text
outputs/noun_phrases/cirr_val_stanza.json
```

Nếu muốn export cả `train` và `test1`:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split all --backend stanza --per_backend_dirs
```

### Bước 2: Audit chất lượng parser

Tạo báo cáo audit:

```powershell
python tools/audit_cirr_phrases.py --input outputs/noun_phrases/cirr_val_stanza.json --output outputs/noun_phrases/cirr_val_stanza_audit.json
```

Chạy gate kiểm tra trước khi train:

```powershell
python tools/check_phrase_audit_gate.py --input outputs/noun_phrases/cirr_val_stanza_audit.json
```

Tạo file sample thủ công tùy chọn để kiểm tra `edit_actions`:

```powershell
python tools/sample_edit_action_audit.py --input outputs/noun_phrases/cirr_val_stanza.json --output outputs/noun_phrases/cirr_val_stanza_edit_actions.csv --n 200
```

### Bước 3: Train hai model

Smoke test nhanh:

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml --fast_dev_run
python -m pytest -q
```

Train `CL-only`:

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl.yaml
```

Best checkpoint:

```text
checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt
```

Train `CL+CC`:

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml --log_cc_grad_norm --cc_grad_norm_interval 200
```

Best checkpoint:

```text
checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt
```

Ghi chú:

* Training chỉ giữ checkpoint tốt nhất (`save_top_k=1`).
* `edit_actions` vẫn chỉ dùng để audit. Chúng không được đưa vào training loss.

### Bước 4: Validate và chọn model thắng

Validate `CL-only`:

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
```

Validate `CL+CC`:

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split val
```

Các metric chính cần so sánh:

```text
recall@1
recall@5
recall@10
recall@50
recall_subset@1
recall_subset@2
recall_subset@3
```

Quy tắc chọn model thắng:

* Nếu `CL+CC` tốt hơn `CL-only` trên các validation metric bạn quan tâm, giữ `CL+CC`.
* Nếu không, giữ `CL-only`.

### Bước 5: Build FAISS indexes

Bạn có thể build FAISS cho cả hai biến thể, hoặc chỉ build cho model thắng.

Build index cho `CL-only`:

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split test1
```

Build index cho `CL+CC`:

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split test1
```

Output thường gặp:

```text
outputs/faiss/cirr_vitb32_cl/cirr_val/
outputs/faiss/cirr_vitb32_cl/cirr_test1/
outputs/faiss/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_val/
outputs/faiss/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_test1/
```

Mỗi thư mục FAISS gồm:

```text
index.faiss
mapping.json
metadata.json
```

## 4. Demo và Visualization

### Bảng port

Dùng port cố định để các app không bị trùng nhau:

| App                      | Port   |
| ------------------------ | ------ |
| Retrieval demo `CL-only` | `8000` |
| Heatmap UI               | `8001` |
| Dataset viewer           | `8002` |
| Retrieval demo `CL+CC`   | `8003` |

### Retrieval demo

Chạy demo `CL-only`:

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --splits val test1 --port 8000
```

Chạy demo `CL+CC`:

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --splits val test1 --port 8003
```

Mở:

```text
CL-only local: http://127.0.0.1:8000
CL-only LAN:   http://<your-lan-ip>:8000
CL+CC local:   http://127.0.0.1:8003
CL+CC LAN:     http://<your-lan-ip>:8003
```

### Heatmap UI

Chạy Heatmap UI với checkpoint rõ ràng:

```powershell
python scripts/generate_loss_heatmaps.py --config configs/cirr_context_cir_cl_cc.yaml --split val --cl_checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --cl_cc_checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --port 8001
```

Mở:

```text
Local: http://127.0.0.1:8001
LAN:   http://<your-lan-ip>:8001
```

Các thư mục output thường gặp:

```text
outputs/heatmaps/cl/
outputs/heatmaps/cl_cc/
outputs/heatmaps/reports/
```

## 5. Submission

Tạo submission cho `CL-only`:

```powershell
python scripts/make_cirr_submission.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --submission_name cirr_vitb32_cl
```

Tạo submission cho `CL+CC`:

```powershell
python scripts/make_cirr_submission.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --submission_name cirr_clip_l14_targetpad_cl_cc_t2i
```

Output:

```text
outputs/submissions/cirr/<submission_name>.json
outputs/submissions/cirr/subset_<submission_name>.json
```

## 6. Thứ tự lệnh chạy nhanh

```powershell
python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir_cl_cc.yaml

python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split val --backend stanza --per_backend_dirs
python tools/audit_cirr_phrases.py --input outputs/noun_phrases/cirr_val_stanza.json --output outputs/noun_phrases/cirr_val_stanza_audit.json
python tools/check_phrase_audit_gate.py --input outputs/noun_phrases/cirr_val_stanza_audit.json

python scripts/train.py --config configs/cirr_context_cir_cl.yaml
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml --log_cc_grad_norm --cc_grad_norm_interval 200

python scripts/validate.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split val

python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --split test1
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split test1

python scripts/retrieve_app.py --config configs/cirr_context_cir_cl.yaml --checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --splits val test1 --port 8000
python scripts/generate_loss_heatmaps.py --config configs/cirr_context_cir_cl_cc.yaml --split val --cl_checkpoint checkpoints/cirr_vitb32_cl/cirr_vitb32_cl-best.ckpt --cl_cc_checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --port 8001
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml --port 8002
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --splits val test1 --port 8003
```

## 7. Các file quan trọng

| Path                                  | Mục đích                            |
| ------------------------------------- | ----------------------------------- |
| `configs/cirr_context_cir_cl.yaml`    | Config cho CL-only                  |
| `configs/cirr_context_cir_cl_cc.yaml` | Config CL+CC chính thức             |
| `scripts/train.py`                    | Entry point để train                |
| `scripts/validate.py`                 | Entry point để validate             |
| `scripts/build_faiss_index.py`        | Build FAISS gallery index           |
| `scripts/retrieve_app.py`             | FastAPI retrieval demo              |
| `scripts/dataset_app.py`              | Dataset viewer                      |
| `scripts/generate_loss_heatmaps.py`   | Heatmap UI                          |
| `scripts/export_noun_phrases.py`      | Export noun phrase                  |
| `tools/audit_cirr_phrases.py`         | Báo cáo audit parser                |
| `tools/check_phrase_audit_gate.py`    | Gate kiểm tra trước khi train CL+CC |

## 8. Khắc phục lỗi thường gặp

### Stanza lỗi trên Windows

Chuẩn bị tài nguyên trước:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --backend stanza --prepare_stanza
```

Nếu vẫn lỗi, đừng train nhầm từ fallback export. Hãy sửa Stanza trước, rồi export lại.

### Retrieval demo mở trước khi FAISS tồn tại

Hãy build FAISS trước cho split bạn muốn xem.

### CUDA out of memory

Giảm các giá trị trong YAML:

```yaml
train:
  batch_size: 32
  val_batch_size: 64
  accumulate_grad_batches: 2
```

hoặc giảm số layer được train:

```yaml
model:
  trainable_vision_layers: 1
  trainable_text_layers: 1
```
