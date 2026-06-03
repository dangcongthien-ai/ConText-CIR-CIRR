from __future__ import annotations

from cir_pipeline.data.transforms import infer_clip_image_size


def test_infer_clip_image_size_handles_rn50x4():
    assert infer_clip_image_size("openai/clip-rn50x4") == 288


def test_infer_clip_image_size_defaults_to_224():
    assert infer_clip_image_size("openai/clip-vit-base-patch32") == 224
