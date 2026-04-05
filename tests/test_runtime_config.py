"""Tests for runtime preset loading and environment config parsing."""

import pytest

from src.configs.environment import EnvironmentManager, RuntimePreset


@pytest.mark.unit
def test_environment_manager_loads_presets():
    manager = EnvironmentManager()
    assert len(manager.presets) >= 5


@pytest.mark.unit
def test_build_environment_config_gpu_24g():
    manager = EnvironmentManager()
    cfg = manager.build_environment_config(RuntimePreset.GPU_24G)

    assert cfg.preset == RuntimePreset.GPU_24G
    assert cfg.device.startswith("cuda")
    assert cfg.max_new_tokens >= 256
    assert cfg.batch_size >= 1


@pytest.mark.unit
def test_get_recommended_models_contains_specialist():
    manager = EnvironmentManager()
    models = manager.get_recommended_models_for_preset(RuntimePreset.GPU_24G)
    assert "chexagent" in models
    assert "radfm" in models
