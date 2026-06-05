# ConText-CIR-CIRR

* backbone: `openai/clip-vit-large-patch14`
* image transform: `targetpad`
* objective: `CL + CC`
* hướng attention của Text-CC: `text -> image`
* pipeline noun phrase: `stanza + constituency_minimal_np`
* config chuẩn: `configs/cirr_context_cir_cl_cc.yaml`
* tên run chuẩn: `cirr_clip_l14_targetpad_cl_cc_t2i`

## 1. Cài đặt

```powershell
cd D:/1_Documents/Nam_3_HK2/Advanced_Computer_Vision/vscode/ConText-CIR-CIRR
conda activate AI
pip install -r requirements.txt
```

Thiết lập Comet tùy chọn:

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

Nếu không muốn sử dụng Comet logging, đặt giá trị sau trong config:

```yaml
logger:
  name: "none"
```

## 2. Cấu trúc dataset mong đợi

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

Mở trình xem dataset:

```powershell
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml --port 8002
```

Truy cập:

```text
Local: http://127.0.0.1:8002
LAN:   http://<your-lan-ip>:8002
```

## 3. Xuất noun phrases

Chuẩn bị tài nguyên Stanza một lần nếu cần:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --backend stanza --prepare_stanza
```

Xuất cache noun phrase chuẩn cho tất cả các split:

```powershell
python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split all --backend stanza
```

Các file cache chuẩn được dùng khi training:

```text
noun_phrases/cirr_train_stanza.json
noun_phrases/cirr_val_stanza.json
noun_phrases/cirr_test1_stanza.json
```

## 4. Audit parser trước khi training

Tạo báo cáo audit:

```powershell
python tools/audit_cirr_phrases.py --input noun_phrases/cirr_val_stanza.json --output outputs/noun_phrases/cirr_val_stanza_audit.json
```

Chạy audit gate:

```powershell
python tools/check_phrase_audit_gate.py --input outputs/noun_phrases/cirr_val_stanza_audit.json
```

Tùy chọn tạo mẫu audit thủ công:

```powershell
python tools/sample_edit_action_audit.py --input noun_phrases/cirr_val_stanza.json --output outputs/noun_phrases/cirr_val_stanza_edit_actions.csv --n 200
```

## 5. Train model chuẩn

Chạy smoke test nhanh:

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml --fast_dev_run
python -m pytest -q
```

Training đầy đủ:

```powershell
python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml
```

Checkpoint tốt nhất:

```text
checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt
```

Ghi chú:

* chỉ giữ checkpoint tốt nhất (`save_top_k=1`)
* `edit_actions` hiện chỉ dùng để audit, chưa được đưa vào training loss
* hướng này sử dụng `text_cc_attention_direction: text_to_image`

## 6. Validate checkpoint đã train

```powershell
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split val
```

Các metric chính cần theo dõi:

```text
recall@1
recall@5
recall@10
recall@50
recall_subset@1
recall_subset@2
recall_subset@3
```

## 7. Build FAISS indexes

Build cho `val`:

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split val
```

Build cho `test1`:

```powershell
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split test1
```

Output mong đợi:

```text
outputs/faiss/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_val/
outputs/faiss/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_test1/
```

Mỗi thư mục FAISS chứa:

```text
index.faiss
mapping.json
metadata.json
```

## 8. Ứng dụng demo

Port map:

* retrieval demo: `8000`
* heatmap UI: `8001`
* dataset viewer: `8002`

### Retrieval demo

```powershell
python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --splits val test1 --port 8000
```

Truy cập:

```text
Local: http://127.0.0.1:8000
LAN:   http://<your-lan-ip>:8000
```

### Dataset viewer

```powershell
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml --port 8002
```

## 9. Submission

```powershell
python scripts/make_cirr_submission.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --submission_name cirr_clip_l14_targetpad_cl_cc_t2i
```

Output mong đợi:

```text
outputs/submissions/cirr/cirr_clip_l14_targetpad_cl_cc_t2i.json
outputs/submissions/cirr/subset_cirr_clip_l14_targetpad_cl_cc_t2i.json
```

## 10. Luồng lệnh nhanh

```powershell
python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir_cl_cc.yaml

python scripts/export_noun_phrases.py --config configs/cirr_context_cir_cl_cc.yaml --split all --backend stanza
python tools/audit_cirr_phrases.py --input noun_phrases/cirr_val_stanza.json --output outputs/noun_phrases/cirr_val_stanza_audit.json
python tools/check_phrase_audit_gate.py --input outputs/noun_phrases/cirr_val_stanza_audit.json

python scripts/train.py --config configs/cirr_context_cir_cl_cc.yaml
python scripts/validate.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split val

python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split val
python scripts/build_faiss_index.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --split test1

python scripts/retrieve_app.py --config configs/cirr_context_cir_cl_cc.yaml --checkpoint checkpoints/cirr_clip_l14_targetpad_cl_cc_t2i/cirr_clip_l14_targetpad_cl_cc_t2i-best.ckpt --splits val test1 --port 8000
python scripts/dataset_app.py --config configs/cirr_context_cir_cl_cc.yaml --port 8002
```

## 11. Các file quan trọng

| Path                                  | Mục đích                                |
| ------------------------------------- | --------------------------------------- |
| `configs/cirr_context_cir_cl_cc.yaml` | config CL+CC chuẩn                      |
| `scripts/train.py`                    | entry point để training                 |
| `scripts/validate.py`                 | entry point để validation               |
| `scripts/build_faiss_index.py`        | công cụ build FAISS index               |
| `scripts/retrieve_app.py`             | demo retrieval bằng FastAPI             |
| `scripts/generate_loss_heatmaps.py`   | heatmap                            |
| `scripts/dataset_app.py`              | trình xem dataset                       |
| `scripts/export_noun_phrases.py`      | xuất noun phrase                        |
| `tools/audit_cirr_phrases.py`         | báo cáo audit parser                    |
| `tools/check_phrase_audit_gate.py`    | gate kiểm tra parser trước khi training |
