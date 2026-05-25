from __future__ import annotations

import math

from cir_pipeline.config import config_to_dict, load_config


def test_default_config_loads():
    cfg = load_config("configs/cirr_context_cir.yaml")
    data = config_to_dict(cfg)
    assert data["project"]["name"] == "CIR_Pipeline"
    assert data["data"]["metadata_root"] == "CIRR/cirr"
    assert data["data"]["stanza_resources_dir"] == ".stanza_resources"
    assert data["data"]["stanza_download_method"] == "none"
    assert data["faiss"]["index_type"] == "flat_ip"
    assert data["model"]["freeze_vision_encoder"] is True
    assert data["model"]["freeze_text_encoder"] is True
    assert data["model"]["trainable_vision_layers"] == 2
    assert data["model"]["trainable_text_layers"] == 2
    assert data["model"]["use_gradient_checkpointing"] is True
    assert math.isclose(data["loss"]["lambda_cont"], 1.0)
    assert math.isclose(data["loss"]["lambda_cc"], 0.2)
    assert math.isclose(data["loss"]["epsilon_cc"], 0.0)
    assert math.isclose(data["loss"]["query_negative_weight"], 0.1)
    assert math.isclose(data["train"]["encoder_learning_rate"], 2.0e-6)
    assert math.isclose(data["train"]["encoder_weight_decay"], 5.0e-3)
    assert data["train"]["batch_size"] == 64
    assert data["train"]["accumulate_grad_batches"] == 1
    assert data["train"]["early_stopping_patience"] == 3
    assert math.isclose(data["train"]["early_stopping_min_delta"], 0.005)
    assert data["train"]["save_top_k"] == 1
    assert data["logger"]["experiment_name"] == "cirr_vitb32_cl_cc"


def test_no_cc_config_loads():
    cfg = load_config("configs/cirr_context_cir_no_cc.yaml")
    data = config_to_dict(cfg)
    assert data["data"]["noun_phrase_backend"] == "none"
    assert math.isclose(data["loss"]["lambda_cont"], 1.0)
    assert math.isclose(data["loss"]["lambda_cc"], 0.0)
    assert data["loss"]["max_noun_phrases"] == 0


def test_variant_configs_load():
    variants = {
        "configs/cirr_context_cir_cl.yaml": ("cirr_vitb32_cl", 1.0, 0.0, "none"),
        "configs/cirr_context_cir_cc.yaml": ("cirr_vitb32_cc", 0.0, 1.0, "stanza"),
        "configs/cirr_context_cir_cl_cc.yaml": ("cirr_vitb32_cl_cc", 1.0, 0.2, "stanza"),
    }
    for path, (experiment_name, lambda_cont, lambda_cc, backend) in variants.items():
        cfg = load_config(path)
        data = config_to_dict(cfg)
        assert data["logger"]["experiment_name"] == experiment_name
        assert math.isclose(data["loss"]["lambda_cont"], lambda_cont)
        assert math.isclose(data["loss"]["lambda_cc"], lambda_cc)
        assert data["data"]["noun_phrase_backend"] == backend
        assert data["train"]["save_top_k"] == 1
        assert data["faiss"]["output_dir"].endswith(experiment_name)
