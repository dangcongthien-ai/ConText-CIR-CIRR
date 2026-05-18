# Data Directory

This project does not store or rewrite CIRR images or metadata here.

Expected local dataset layout:

```text
CIRR/
  train/
  dev/
  test1/
  cirr/
    image_splits/split.rc2.*.json
    captions/cap.rc2.*.json
    captions_ext/cap.ext.rc2.*.json
```

Training, validation, indexing, and retrieval resolve image paths only from
`image_splits/split.rc2.*.json`. The image folders may contain extra PNG files;
those extras are ignored by the data pipeline.
