from __future__ import annotations

import math

from cir_pipeline.config import config_to_dict, load_config


def test_default_config_loads():
    cfg = load_config("configs/cirr_context_cir.yaml")
    data = config_to_dict(cfg)
    assert data["project"]["name"] == "CIR_Pipeline"
    assert data["data"]["metadata_root"] == "CIRR/cirr"
    assert data["faiss"]["index_type"] == "flat_ip"
    assert data["model"]["freeze_vision_encoder"] is True
    assert data["model"]["freeze_text_encoder"] is True
    assert data["model"]["trainable_vision_layers"] == 2
    assert data["model"]["trainable_text_layers"] == 2
    assert data["model"]["use_gradient_checkpointing"] is True
    assert math.isclose(data["loss"]["query_negative_weight"], 0.1)
    assert math.isclose(data["train"]["encoder_learning_rate"], 2.0e-6)
    assert math.isclose(data["train"]["encoder_weight_decay"], 5.0e-3)
    assert data["train"]["batch_size"] == 64
    assert data["train"]["accumulate_grad_batches"] == 1
    assert data["train"]["early_stopping_patience"] == 3
    assert math.isclose(data["train"]["early_stopping_min_delta"], 0.005)
