# Test Suite Documentation

## Overview

The test suite is organized by component with clear markers for categorization and easy filtering.

**Total Coverage:**
- **Model Registry Tests** (test_registry.py): 14 tests
- **Model Tests** (test_models.py): Various wrapper tests
- **Evaluation Tests** (test_evaluation.py): Metric and evaluator tests

## Structure

```
tests/
├── conftest.py           # Pytest fixtures and configuration
├── __init__.py           # Package marker
├── test_registry.py      # ✓ NEW: Model registry tests (14 tests)
├── test_models.py        # Model wrapper tests
├── test_evaluation.py    # Evaluation metric tests
└── README.md             # This file
```

## Test Markers

Tests are organized with markers for easy filtering:

| Marker | Purpose | Examples |
|--------|---------|----------|
| `@pytest.mark.unit` | Fast, isolated unit tests | Model instantiation, utility functions |
| `@pytest.mark.integration` | Tests combining multiple components | Full pipeline testing |
| `@pytest.mark.slow` | Potentially slow tests (> 10 seconds) | Model loading, dataset loading |
| `@pytest.mark.models` | Tests for model wrappers | Model-specific tests |
| `@pytest.mark.evaluation` | Tests for evaluation metrics | BLEU, RadGraph, IoU tests |
| `@pytest.mark.data` | Tests for data loaders | Dataset tests |
| `@pytest.mark.registry` | Tests for model registry | NEW: Registry consistency tests |
| `@pytest.mark.skip_ci` | Skip in CI/CD pipelines | Tests requiring GPU, API keys |

## Running Tests

### Install pytest (if not installed)
```bash
pip install pytest pytest-timeout
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories

#### Registry Tests (Fast - 1-2 seconds)
```bash
pytest tests/test_registry.py -v
# Tests the 14 MAIN + 7 EXTRA model configuration
```

#### Unit Tests Only (Fast - 5-10 seconds)
```bash
pytest tests/ -m unit -v
```

#### Model Tests Only
```bash
pytest tests/test_models.py -v -m models
```

#### Evaluation Tests Only
```bash
pytest tests/test_evaluation.py -v -m evaluation
```

#### Skip Slow Tests
```bash
pytest tests/ -v -m "not slow"
```

#### Run Only Registry Tests (Validates model_registry.py changes)
```bash
pytest tests/test_registry.py -v -m registry
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
# Coverage report will be in htmlcov/index.html
```

### Run with Detailed Output
```bash
pytest tests/ -vvv --tb=long
```

### Run Tests in Parallel (faster)
```bash
pip install pytest-xdist
pytest tests/ -n auto  # Uses all CPU cores
```

## Key Test Categories

### 1. Registry Tests (NEW - test_registry.py)

Validates the Week 1 refactoring work:

- ✅ 14 MAIN benchmark models registered
- ✅ 7 EXTRA track models registered
- ✅ No overlap between MAIN and EXTRA
- ✅ Total: 21 models
- ✅ Gated models properly flagged (Llama-3.2, CheXagent)
- ✅ Access validation functions work

**Run:** `pytest tests/test_registry.py -v`

### 2. Model Tests (test_models.py)

Tests individual model wrappers:

- Model initialization
- Configuration loading
- Report parsing
- Output formatting

### 3. Evaluation Tests (test_evaluation.py)

Tests evaluation metrics:

- BLEU, ROUGE scoring
- RadGraph F1 calculation
- Bounding box IoU computation
- Statistical testing

## Quick Test Sequence

For a quick validation run:

```bash
# 1. Fast smoke test (registry + unit)
pytest tests/test_registry.py tests/test_evaluation.py -m unit -v

# 2. Review model loading (requires GPU, slower)
pytest tests/test_models.py -v -m "not slow" 

# 3. Full suite (comprehensive)
pytest tests/ -v --tb=short
```

## Understanding Test Output

### Green (✓)
Test passed successfully.

### Red (✗)
Test failed - review the `FAILED` line for location and error.

### Yellow (⚠)
Test had warnings - check output but may still pass.

### Skipped (S)
Test was skipped (marked with `@pytest.mark.skip` or marked for CI-only).

## Fixtures Available

Tests can use these fixtures (defined in conftest.py):

```python
def test_something(
    project_root_path,        # Path to project root
    config_dir,               # Path to configs/ 
    model_config_dir,         # Path to configs/model_configs/
    sample_image_array,       # Random 512x512 RGB image
    sample_report_text,       # Sample medical report
    model_registry,           # Loaded registry data
    mock_model_config,        # Sample config dict
):
    pass
```

## Troubleshooting

### "No module named 'pytest'"
```bash
pip install pytest pytest-timeout
```

### "No module named 'src'"
Make sure you're running pytest from the project root:
```bash
cd path/to/LLMComparison
pytest tests/
```

### Tests hang or timeout
Some tests require GPU. Skip them:
```bash
pytest tests/ -m "not slow"
```

### Specific test fails on your machine
Check environment setup:
```bash
# Verify model registry
python -c "from src.utils.model_registry import get_main_benchmark_models; print(len(get_main_benchmark_models()))"
# Should output: 14
```

## Adding New Tests

When adding new tests:

1. **Follow naming:** `test_*.py` files, `test_*` functions/methods, `Test*` classes
2. **Add markers:** Use `@pytest.mark.appropriate_marker` 
3. **Use fixtures:** Leverage fixtures from conftest.py
4. **Document:** Add docstrings explaining what's being tested

Example:

```python
@pytest.mark.evaluation
@pytest.mark.unit
def test_new_metric(model_registry, sample_image_array):
    """Test that new metric computes correctly."""
    from src.evaluation.new_metric import NewMetric
    
    metric = NewMetric()
    result = metric.compute(sample_image_array)
    
    assert result > 0
    assert "score" in result
```

## CI/CD Integration

For GitHub Actions or similar:

```yaml
- name: Run Tests
  run: |
    pip install pytest pytest-timeout pytest-cov
    pytest tests/ -m "not skip_ci" --cov=src --junitxml=test-results.xml
```

## Progress Tracking

| Component | Tests | Last Run | Status |
|-----------|-------|----------|--------|
| Registry | 14 | ✅ This session | PASS (14/14) |
| Models | ~10 | Pending | - |
| Evaluation | ~8 | Pending | - |
| **Total** | **32+** | **TBD** | - |

---

Last updated: Week 1 completion - Registry tests added for model_registry.py refactoring
