"""Pytest configuration and fixtures.

Provides common fixtures and setup for all tests.
"""

import pytest
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Get project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_dir(project_root_path):
    """Get config directory path."""
    return project_root_path / "configs"


@pytest.fixture(scope="session")
def model_config_dir(config_dir):
    """Get model config directory."""
    return config_dir / "model_configs"


@pytest.fixture
def sample_image_array():
    """Create a sample image array for testing."""
    import numpy as np
    
    # Create a simple 512x512 RGB image
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_report_text():
    """Create a sample medical report for testing."""
    return """
FINDINGS:
The heart size is normal. The lungs are clear bilaterally.
No pneumothorax or pleural effusion. The mediastinal contours are normal.

IMPRESSION:
No acute cardiopulmonary process.
"""


@pytest.fixture(scope="session")
def model_registry():
    """Load model registry once for all tests."""
    from src.utils.model_registry import MODEL_REGISTRY, get_main_benchmark_models, get_extra_track_models
    
    return {
        "registry": MODEL_REGISTRY,
        "main_models": get_main_benchmark_models(),
        "extra_models": get_extra_track_models(),
    }


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    return {
        "name": "test-model",
        "type": "local",
        "provider": "huggingface",
        "model_path": "test/test-model",
        "loading": {
            "device_map": "auto",
            "torch_dtype": "float16",
            "load_in_4bit": False,
        }
    }


# Pytest hooks for test collection and reporting

def pytest_collection_modifyitems(config, items):
    """Customize test collection."""
    for item in items:
        # Add default markers if not present
        if "slow" not in item.keywords and "skip_ci" not in item.keywords:
            # Mark integration and data tests as potentially slow
            if "integration" in item.keywords or "data" in item.keywords:
                item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    """Configure pytest before test run."""
    # Add custom ini options
    config.addinivalue_line(
        "markers",
        "smoke: Quick smoke tests for basic functionality"
    )


# Test collection helpers

def pytest_generate_tests(metafunc):
    """Generate parameterized tests."""
    
    # Parametrize tests that work with benchmark tiers
    if "benchmark_tier" in metafunc.fixturenames:
        from src.utils.model_registry import BenchmarkTier
        
        tiers = [BenchmarkTier.MAIN, BenchmarkTier.EXTRA]
        metafunc.parametrize("benchmark_tier", tiers, ids=["main", "extra"])
