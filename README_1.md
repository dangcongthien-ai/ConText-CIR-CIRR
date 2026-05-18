# CIR_Pipeline: Truy hồi CIRR kiểu ConText-CIR

Pipeline PyTorch Lightning chuyên nghiệp cho bài toán Composed Image Retrieval
trên CIRR. Model đi theo ý tưởng ConText-CIR: mã hóa ảnh truy vấn và văn bản
chỉnh sửa, fuse visual tokens và text tokens bằng cross-attention, pooling thành
retrieval embedding chung, rồi train bằng contrastive retrieval loss cùng Text
Concept-Consistency loss tùy chọn.

## Kiến Trúc

Training:

1. Nhánh query tính `r_q = f(I_q, T)` từ CLIP visual tokens, CLIP text tokens,
   cross-attention, và attention pooling.
2. Nhánh target tính `r_t = f(I_t, "")`, dùng đúng cùng fusion network với blank
   text.
3. Nhánh query-negative tính `r_q_neg = f(I_q, "")`.
4. Loss là `L_total = L_cont + lambda_cc * L_CC`, trong đó `L_cont` là InfoNCE
   trên in-batch targets cộng query-image negatives, còn `L_CC` so sánh attention
   map giữa full-caption và isolated noun-phrase.

Inference:

1. Mã hóa mọi ảnh gallery chính thức dưới dạng `f(C_i, "")`.
2. L2-normalize embeddings và lưu vào FAISS `IndexFlatIP` theo mặc định.
3. Mã hóa query dưới dạng `f(I_q, T)`, search FAISS, và loại ảnh reference khỏi
   kết quả trả về nếu ảnh đó xuất hiện trong gallery.

## Cấu Trúc Dataset

Cấu trúc local mong đợi:

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

Pipeline không bao giờ train, validate, index, hoặc retrieve bằng cách scan thư
mục ảnh. File ảnh chỉ được resolve thông qua `image_splits/split.rc2.*.json`.
Điều này quan trọng vì các thư mục ảnh local có thể là superset, chứa thêm PNG
không thuộc annotation CIRR rc2.

Các số liệu metadata local được dùng làm kỳ vọng khi inspect:

| Split | Ảnh chính thức | Cặp caption | Ghi chú |
|---|---:|---:|---|
| train | 16,939 | 28,225 | ảnh train nằm trong các thư mục con |
| val | 2,297 | 4,181 | metadata `val` ánh xạ tới local `dev` |
| test1 | 2,315 | 4,148 | một caption rỗng được xử lý an toàn |

## Cài Đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu muốn dùng FAISS GPU, hãy cài bản `faiss-gpu` khớp với CUDA của bạn thay vì
`faiss-cpu`.

## COMET

Không hardcode API key. Cách dễ nhất là copy file env mẫu:

```bash
copy .env.example .env
```

Sau đó sửa `.env`:

```bash
COMET_API_KEY=your_key
COMET_WORKSPACE=your_workspace
COMET_PROJECT_NAME=ConText-CIR-CIRR
```

Project sẽ tự load `.env` từ root trước khi tạo COMET logger. Bạn cũng có thể
set biến môi trường thủ công:

```bash
set COMET_API_KEY=your_key
set COMET_WORKSPACE=your_workspace
set COMET_PROJECT_NAME=ConText-CIR-CIRR
```

Nếu `logger.name=comet` nhưng thiếu `COMET_API_KEY`, training sẽ in warning và
tiếp tục chạy thay vì crash. Bạn cũng có thể đặt `logger.name: "tensorboard"`
hoặc `logger.name: "none"` trong YAML.

## Lệnh Chạy

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

Tạo submission CIRR test1:

```bash
python scripts/make_cirr_submission.py --config configs/cirr_context_cir.yaml --checkpoint checkpoints/context_cir_cirr_baseline/best.ckpt
```

## Tác Dụng File Và Thư Mục

| Path | Tác dụng |
|---|---|
| `configs/cirr_context_cir.yaml` | Config đầy đủ cho train/eval/retrieval/logger/path. |
| `src/cir_pipeline/config.py` | Load và validate config bằng dataclasses. |
| `src/cir_pipeline/seed.py` | Seed Python, NumPy, và Torch. |
| `src/cir_pipeline/data/cirr_dataset.py` | CIRR Dataset chỉ đọc official image_splits cộng captions. |
| `src/cir_pipeline/data/cirr_datamodule.py` | Lightning DataModule cho train/val/test dataloaders. |
| `src/cir_pipeline/data/transforms.py` | Transform ảnh resize/crop/normalize chuẩn CLIP. |
| `src/cir_pipeline/data/collate.py` | Collate image tensors, captions, noun phrases, pair ids, names, và img_set. |
| `src/cir_pipeline/data/validate_cirr_files.py` | Inspect metadata/tình trạng ảnh ở chế độ read-only. |
| `src/cir_pipeline/models/encoders.py` | HuggingFace CLIP image và text token encoders. |
| `src/cir_pipeline/models/attention.py` | Multi-head cross-attention và learned attention pooling. |
| `src/cir_pipeline/models/context_cir.py` | Model chính `ConTextCIR` với `forward_query`, `encode_gallery`, và attention output. |
| `src/cir_pipeline/models/losses.py` | InfoNCE retrieval loss và Text Concept-Consistency loss. |
| `src/cir_pipeline/text/noun_phrase_parser.py` | Trích xuất noun phrase bằng Stanza, NLTK, hoặc regex fallback. |
| `src/cir_pipeline/text/token_alignment.py` | Map noun phrase sang CLIP token index an toàn bằng tokenizer offsets. |
| `src/cir_pipeline/engine/train_module.py` | LightningModule cho train/validation step, optimizer, và scheduler. |
| `src/cir_pipeline/engine/metrics.py` | Recall@K, RecallSubset@K, và similarity utilities. |
| `src/cir_pipeline/engine/evaluator.py` | Evaluator full-gallery validation retrieval. |
| `src/cir_pipeline/engine/comet_logger.py` | Tạo COMET/TensorBoard logger và helper log artifact. |
| `src/cir_pipeline/retrieval/faiss_index.py` | Wrapper tạo/lưu/load/search FAISS. |
| `src/cir_pipeline/retrieval/build_gallery.py` | Encode gallery images bằng `f(C_i, "")` và lưu FAISS artifacts. |
| `src/cir_pipeline/retrieval/search.py` | Encode query `f(I_q,T)`, search FAISS, và loại reference image. |
| `src/cir_pipeline/utils/io.py` | Helper JSON và filesystem. |
| `src/cir_pipeline/utils/device.py` | Helper đưa batch lên device theo kiểu đệ quy. |
| `src/cir_pipeline/utils/logging.py` | Helper setup logging. |
| `scripts/train.py` | CLI entrypoint để train. |
| `scripts/validate.py` | CLI validate checkpoint. |
| `scripts/build_faiss_index.py` | CLI build FAISS index. |
| `scripts/retrieve.py` | CLI demo retrieval bằng ảnh cộng text. |
| `scripts/make_cirr_submission.py` | Tạo JSON submission CIRR test1 cho recall và recall_subset. |
| `scripts/inspect_cirr_dataset.py` | CLI inspect dataset. |
| `tests/` | Unit tests cho dataset, losses, và FAISS. |
| `data/README.md` | Ghi chú vị trí dataset; không lưu raw data ở đây. |
| `outputs/.gitkeep` | Giữ thư mục output sinh ra trong project tree. |
| `checkpoints/.gitkeep` | Giữ thư mục checkpoint trong project tree. |

## Troubleshooting

Thiếu `COMET_API_KEY`:
Set biến môi trường COMET, dùng `logger.offline: true`, hoặc chuyển sang
`logger.name: "none"`.

Thiếu Stanza model:
Parser sẽ warning và fallback sang NLTK hoặc regex. Để trích xuất noun phrase
tốt hơn, chạy `python -c "import stanza; stanza.download('en')"`.

CUDA OOM:
Giảm `train.batch_size`, `train.val_batch_size`, hoặc dùng gradient
accumulation bằng cách thêm cấu hình đó vào Trainer trong `scripts/train.py`.

Lỗi cài FAISS:
Dùng `pip install faiss-cpu` trên máy CPU. Với CUDA, hãy tự cài package
`faiss-gpu` tương thích.

Thiếu image path:
Chạy `python scripts/inspect_cirr_dataset.py --config configs/cirr_context_cir.yaml`.
Loader yêu cầu mọi image path khai báo trong `image_splits` phải tồn tại dưới
`data.root`.
