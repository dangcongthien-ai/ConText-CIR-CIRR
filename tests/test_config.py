from __future__ import annotations

from cir_pipeline.config import config_to_dict, load_config


def test_default_config_loads():
    cfg = load_config("configs/cirr_context_cir.yaml")
    data = config_to_dict(cfg)
    assert data["project"]["name"] == "CIR_Pipeline"
    assert data["data"]["metadata_root"] == "CIRR/cirr"
    assert data["faiss"]["index_type"] == "flat_ip"
