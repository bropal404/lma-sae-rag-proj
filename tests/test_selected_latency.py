"""Regression tests for selected-model H4 routing and stage composition."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _module():
    script = Path(__file__).resolve().parents[1] / "scripts/sagemaker/latency_selected_route.py"
    spec = spec_from_file_location("latency_selected_route", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_layer_names_and_baselines():
    f = _module()
    sae, dense, routers = f.selected_routers("gemma3-4b-it", "miss_as_retrieve", 29, 22)
    assert sae.endswith("sae16k_last@L29__l2")
    assert dense.endswith("resid_last@L22__l2")
    assert dense in routers and sae in routers
    assert "b1_always_retrieve" in routers
    assert "resid_last@L29" not in " ".join(routers)


def test_action_stage_costs():
    f = _module()
    stage = {"sae_controller_ms": 70, "dense_controller_ms": 45,
             "closed_ms": 200, "rag_ms": 225, "bm25_ms": 15, "b4_ms": 300}
    assert f.cost_ms("ANSWER", stage, "sae_controller_ms") == 270
    assert f.cost_ms("RETRIEVE", stage, "sae_controller_ms") == 310
    assert f.cost_ms("ABSTAIN", stage, "sae_controller_ms") == 70
    assert f.cost_ms("ANSWER", stage, "dense_controller_ms") == 245
    assert f.cost_ms("RETRIEVE", stage, None) == 240
    assert f.cost_ms("ANSWER", stage, "b4_ms") == 500
    with pytest.raises(ValueError):
        f.cost_ms("INVALID", stage, "sae_controller_ms")
