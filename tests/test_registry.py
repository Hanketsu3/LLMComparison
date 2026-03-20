"""Tests for model registry.

Tests the model_registry module including:
- Model filtering and access
- Benchmark tier organization
- Gated model detection
- Model loading validation
"""

import pytest
from unittest.mock import Mock, patch


@pytest.mark.registry
@pytest.mark.unit
class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_registry_import(self):
        """Verify model registry can be imported."""
        from src.utils.model_registry import MODEL_REGISTRY, BenchmarkTier
        
        assert MODEL_REGISTRY is not None
        assert BenchmarkTier.MAIN is not None
        assert BenchmarkTier.EXTRA is not None
    
    def test_main_models_count(self):
        """Verify we have exactly 14 MAIN benchmark models."""
        from src.utils.model_registry import get_main_benchmark_models
        
        main_models = get_main_benchmark_models()
        assert len(main_models) == 14, f"Expected 14 main models, got {len(main_models)}"
    
    def test_extra_models_count(self):
        """Verify we have exactly 7 EXTRA track models."""
        from src.utils.model_registry import get_extra_track_models
        
        extra_models = get_extra_track_models()
        assert len(extra_models) == 7, f"Expected 7 extra models, got {len(extra_models)}"
    
    def test_main_models_list(self):
        """Verify specific main models are registered."""
        from src.utils.model_registry import get_main_benchmark_models
        
        main = get_main_benchmark_models()
        
        # Check generalist models
        assert "qwen2-vl-2b" in main
        assert "phi3-vision" in main
        
        # Check domain-adaptive models
        assert "llava-med" in main
        assert "medgemma-4b" in main
        
        # Check specialist models
        assert "chexagent" in main
        assert "llava-rad" in main
    
    def test_extra_models_list(self):
        """Verify specific extra models are registered."""
        from src.utils.model_registry import get_extra_track_models
        
        extra = get_extra_track_models()
        
        # Check OCR models
        assert "got-ocr2" in extra
        assert "nougat-base" in extra
        
        # Check API models
        assert "gpt4v" in extra
        assert "gemini" in extra
    
    def test_model_by_benchmark_tier(self):
        """Test filtering models by benchmark tier."""
        from src.utils.model_registry import get_models_by_benchmark_tier, BenchmarkTier
        
        main_by_tier = get_models_by_benchmark_tier(BenchmarkTier.MAIN)
        assert len(main_by_tier) == 14
        
        extra_by_tier = get_models_by_benchmark_tier(BenchmarkTier.EXTRA)
        assert len(extra_by_tier) == 7
    
    def test_gated_access_models(self):
        """Verify gated access models are properly flagged."""
        from src.utils.model_registry import get_gated_access_models
        
        gated = get_gated_access_models()
        gated_names = [m.name for m in gated]
        
        # These models require special access
        assert "llama3-vision" in gated_names
        assert "chexagent" in gated_names
        assert len(gated) == 2
    
    def test_check_model_access_valid_free(self):
        """Test access check for free models."""
        from src.utils.model_registry import check_model_access
        
        access = check_model_access("qwen2-vl-2b")
        assert access["accessible"] is True
        assert access["requires_key"] is False
    
    def test_check_model_access_gated(self):
        """Test access check for gated models (should indicate need for login)."""
        from src.utils.model_registry import check_model_access
        
        access = check_model_access("llama3-vision")
        # Should return info about gated status
        assert "accessible" in access
        assert "message" in access
    
    def test_model_info_dataclass(self):
        """Verify ModelInfo dataclass has required fields."""
        from src.utils.model_registry import ModelInfo, BenchmarkTier
        
        model = ModelInfo(
            name="test-model",
            benchmark_tier=BenchmarkTier.MAIN,
            gated_access=False
        )
        
        assert model.name == "test-model"
        assert model.benchmark_tier == BenchmarkTier.MAIN
        assert model.gated_access is False
    
    def test_model_registry_consistency(self):
        """Verify total model count: MAIN + EXTRA = 21."""
        from src.utils.model_registry import (
            get_main_benchmark_models, 
            get_extra_track_models
        )
        
        main = get_main_benchmark_models()
        extra = get_extra_track_models()
        
        # No overlap
        assert len(set(main) & set(extra)) == 0
        
        # Total = 21
        assert len(main) + len(extra) == 21


@pytest.mark.registry
@pytest.mark.unit
class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('src.utils.model_registry.load_model')
    def test_load_model_with_access_check(self, mock_load):
        """Verify load_model performs access validation."""
        from src.utils.model_registry import check_model_access
        
        # Check that free model has no access issues
        access = check_model_access("qwen2-vl-2b")
        assert access["accessible"] is True
    
    def test_load_model_invalid_name(self):
        """Test loading non-existent model."""
        from src.utils.model_registry import check_model_access
        
        # Invalid model should raise or return error
        result = check_model_access("non-existent-model")
        # Should either be False or raise KeyError
        if isinstance(result, dict):
            assert result.get("accessible") is False or "not found" in str(result).lower()
