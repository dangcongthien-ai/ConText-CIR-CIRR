from __future__ import annotations

import math

from cir_pipeline.config import config_to_dict, load_config


def test_default_cl_cc_config_loads_as_best_candidate():
    cfg = load_config("configs/cirr_context_cir_cl_cc.yaml")
    data = config_to_dict(cfg)
    assert data["project"]["name"] == "CIR_Pipeline"
    assert data["data"]["metadata_root"] == "CIRR/cirr"
    assert data["data"]["noun_phrase_cache_dir"] == "noun_phrases"
    assert data["data"]["use_exported_noun_phrases"] is True
    assert data["data"]["stanza_resources_dir"] == ".stanza_resources"
    assert data["data"]["stanza_download_method"] == "none"
    assert data["data"]["noun_phrase_backend"] == "stanza"
    assert data["data"]["noun_phrase_extractor"] == "constituency_minimal_np"
    assert data["faiss"]["index_type"] == "flat_ip"
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
    assert math.isclose(data["train"]["encoder_learning_rate"], 2.0e-6)
    assert math.isclose(data["train"]["encoder_weight_decay"], 5.0e-3)
    assert data["train"]["batch_size"] == 64
    assert data["train"]["accumulate_grad_batches"] == 1
    assert data["train"]["early_stopping_patience"] == 3
    assert math.isclose(data["train"]["early_stopping_min_delta"], 0.005)
    assert data["train"]["save_top_k"] == 1
    assert data["logger"]["experiment_name"] == "cirr_vitb32_cl_cc_best"


def test_only_two_canonical_training_configs_remain():
    variants = {
        "configs/cirr_context_cir_cl.yaml": {
            "experiment_name": "cirr_vitb32_cl",
            "lambda_cont": 1.0,
            "lambda_cc": 0.0,
            "backend": "none",
            "extractor": "constituency",
            "clean_only": False,
        },
        "configs/cirr_context_cir_cl_cc.yaml": {
            "experiment_name": "cirr_vitb32_cl_cc_best",
            "lambda_cont": 1.0,
            "lambda_cc": 0.08,
            "backend": "stanza",
            "extractor": "constituency_minimal_np",
            "clean_only": True,
        },
    }
    for path, expected in variants.items():
        cfg = load_config(path)
        data = config_to_dict(cfg)
        assert data["logger"]["experiment_name"] == expected["experiment_name"]
        assert math.isclose(data["loss"]["lambda_cont"], expected["lambda_cont"])
        assert math.isclose(data["loss"]["lambda_cc"], expected["lambda_cc"])
        assert data["data"]["noun_phrase_backend"] == expected["backend"]
        assert data["data"]["noun_phrase_extractor"] == expected["extractor"]
        assert data["data"]["noun_phrase_cache_dir"] == "noun_phrases"
        assert data["data"]["use_exported_noun_phrases"] is True
        assert data["loss"]["text_cc_clean_only"] is expected["clean_only"]
        assert math.isclose(data["loss"]["text_cc_clean_min_confidence"], 0.7)
        assert data["loss"]["text_cc_attention_direction"] == "text_to_image"
        assert data["train"]["save_top_k"] == 1
        assert data["train"]["log_cc_grad_norm"] is False
        assert data["train"]["cc_grad_norm_interval"] == 0
        assert data["faiss"]["output_dir"].endswith(expected["experiment_name"])


def test_experimental_clipmodel_config_enables_native_projection():
    cfg = load_config("configs/cirr_context_cir_cl_clipmodel.yaml")
    data = config_to_dict(cfg)
    assert data["model"]["use_native_clip_projection"] is True
    assert data["logger"]["experiment_name"] == "cirr_vitb32_cl_clipmodel"
    assert data["faiss"]["output_dir"].endswith("cirr_vitb32_cl_clipmodel")


def test_targetpad_config_loads_with_expected_transform_settings():
    cfg = load_config("configs/cirr_context_cir_cl_targetpad.yaml")
    data = config_to_dict(cfg)
    assert data["data"]["image_transform"] == "targetpad"
    assert math.isclose(data["data"]["target_pad_ratio"], 1.25)
    assert data["logger"]["experiment_name"] == "cirr_vitb32_cl_targetpad"
    assert data["faiss"]["output_dir"].endswith("cirr_vitb32_cl_targetpad")
