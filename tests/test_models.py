"""Tests for model modules."""

import pytest
from unittest.mock import Mock, patch


class TestBaseModel:
    """Test base model functionality."""
    
    def test_model_output_dataclass(self):
        from src.models.base_model import ModelOutput
        
        output = ModelOutput(
            text="Test report",
            findings="Test findings",
            impression="Test impression"
        )
        
        assert output.text == "Test report"
        assert output.findings == "Test findings"
    
    def test_base_model_abstract(self):
        from src.models.base_model import BaseRadiologyModel
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseRadiologyModel("test")


class TestGPT4V:
    """Test GPT-4V model wrapper."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_gpt4v_init(self):
        from src.models.generalist.gpt4v import GPT4VModel
        
        model = GPT4VModel(model_name="gpt-4-vision-preview")
        assert model.model_name == "gpt-4-vision-preview"
        assert model.model_type == "api"
    
    def test_gpt4v_parse_report(self):
        from src.models.generalist.gpt4v import GPT4VModel
        
        model = GPT4VModel()
        
        text = """FINDINGS: The heart is normal size.
        
IMPRESSION: No acute cardiopulmonary process."""
        
        findings, impression = model._parse_report(text)
        
        assert "heart is normal size" in findings
        assert "No acute" in impression


class TestCheXagent:
    """Test CheXagent model wrapper."""
    
    def test_chexagent_init(self):
        from src.models.specialist.chexagent import CheXagentModel
        
        model = CheXagentModel()
        assert model.model_name == "StanfordAIMI/CheXagent-8b"
        assert model.supports_grounding == True
    
    def test_chexagent_parse_bbox(self):
        from src.models.specialist.chexagent import CheXagentModel
        
        model = CheXagentModel()
        
        text = "The finding is located at [10, 20, 100, 150]"
        bboxes = model._parse_bounding_boxes(text)
        
        assert len(bboxes) == 1
        assert bboxes[0]["x_min"] == 10
        assert bboxes[0]["y_max"] == 150
