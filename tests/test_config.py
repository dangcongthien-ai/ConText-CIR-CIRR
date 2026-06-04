from __future__ import annotations

import math

from cir_pipeline.config import config_to_dict, load_config


def test_canonical_cl_cc_config_loads():
    cfg = load_config("configs/cirr_context_cir_cl_cc.yaml")
    data = config_to_dict(cfg)

    assert data["project"]["name"] == "CIR_Pipeline"
    assert data["data"]["dataset"] == "cirr"
    assert data["data"]["noun_phrase_backend"] == "stanza"
    assert data["data"]["noun_phrase_extractor"] == "constituency_minimal_np"
    assert data["data"]["noun_phrase_cache_dir"] == "noun_phrases"
    assert data["data"]["use_exported_noun_phrases"] is True
    assert data["data"]["image_transform"] == "targetpad"
    assert math.isclose(data["data"]["target_pad_ratio"], 1.25)

    assert data["model"]["backbone"] == "openai/clip-vit-large-patch14"
    assert data["model"]["backbone_size"] == "L"
    assert data["model"]["freeze_vision_encoder"] is True
    assert data["model"]["freeze_text_encoder"] is True
    assert data["model"]["trainable_vision_layers"] == 2
    assert data["model"]["trainable_text_layers"] == 2
    assert data["model"]["use_gradient_checkpointing"] is True

    assert math.isclose(data["loss"]["lambda_cont"], 1.0)
    assert math.isclose(data["loss"]["lambda_cc"], 0.08)
    assert math.isclose(data["loss"]["epsilon_cc"], 0.0)
    assert math.isclose(data["loss"]["query_negative_weight"], 0.1)
    assert data["loss"]["text_cc_clean_only"] is True
    assert math.isclose(data["loss"]["text_cc_clean_min_confidence"], 0.7)
    assert data["loss"]["text_cc_attention_direction"] == "text_to_image"

    assert data["train"]["batch_size"] == 16
    assert data["train"]["val_batch_size"] == 32
    assert data["train"]["accumulate_grad_batches"] == 2
    assert math.isclose(data["train"]["learning_rate"], 1.5e-5)
    assert math.isclose(data["train"]["encoder_learning_rate"], 1.0e-6)
    assert math.isclose(data["train"]["encoder_weight_decay"], 5.0e-3)
    assert data["train"]["save_top_k"] == 1

    assert data["logger"]["experiment_name"] == "cirr_clip_l14_targetpad_cl_cc_t2i"
    assert data["faiss"]["output_dir"].endswith("cirr_clip_l14_targetpad_cl_cc_t2i")
