# Multi-Dataset Support Architecture

## Executive Summary

Enable running the same 14 MAIN benchmark models across multiple datasets (MIMIC-CXR, VQA-RAD, MS-CXR, IU X-Ray) with unified results aggregation, cross-dataset statistical testing, and dataset-specific performance analysis.

**Goal:** Answer "How does model X perform across different radiology datasets?"

---

## Current State vs. Proposed State

### ❌ Current (Single Dataset)
- `run_comparison.py` loads **ONE** dataset (e.g., VQA-RAD)
- Run once per dataset manually
- Results stored separately
- No cross-dataset comparison

### ✅ Proposed (Multi-Dataset)
- Load **MULTIPLE** datasets in parallel/sequential pipelines
- Single unified experiment run
- Automatic result aggregation
- Cross-dataset statistical analysis (ANOVA, effect sizes)
- Per-dataset performance tables

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│           Multi-Dataset Experiment Framework                 │
└─────────────────────────────────────────────────────────────┘

INPUT:
├── Model Selection: get_main_benchmark_models()  [14 models]
├── Dataset Selection: MultiDatasetLoader       [4 datasets]
├── Task Configuration: RRG, VQA, Grounding
└── Evaluation Metrics: BLEU, RadGraph, VQA Acc

CORE PROCESSING:
┌─────────────────────────────────────────────────────────────┐
│ for each dataset:                                            │
│   for each model:                                            │
│     load_model(model_name)                                  │
│     for each task:                                           │
│       predictions = model.predict(dataset_samples)          │
│       metrics = evaluate_predictions(task, dataset)         │
│       save_results(model, dataset, task, metrics)           │
│                                                              │
│ after all datasets:                                          │
│   aggregate_results()  → cross_dataset_table               │
│   statistical_tests()  → significance analysis              │
│   generate_report()    → findings                           │
└─────────────────────────────────────────────────────────────┘

OUTPUT:
├── per_model_per_dataset_metrics.json
├── aggregate_comparison_table.csv
├── statistical_significance.json
├── dataset_specific_analysis.md
└── cross_dataset_plots.png
```

---

## Dataset Specifications

| Dataset | Train | Test | Task | Size | Key Features |
|---------|-------|------|------|------|--------------|
| **MIMIC-CXR** (default) | 354K | 45K | RRG, VQA | Large | Diverse, varied findings |
| **VQA-RAD** | 3,500 | 665 | VQA | Medical QA | Domain-specific Q&A |
| **MS-CXR** | 2,000 | 500 | RRG, VQA | Multilingual | Multi-lingual reports |
| **IU X-Ray** | 2,800 | 550 | RRG | Small | Lower complexity |

**Multi-task Coverage:**
- ✅ Report Generation (RRG): MIMIC-CXR, MS-CXR, IU X-Ray
- ✅ Visual QA (VQA): VQA-RAD, MIMIC-CXR (custom QA)
- ✅ Grounding: Any (if model supports)

---

## Implementation Plan

### Phase 1: Data Loader (Week 2, 1 hour)

**File:** `src/data/multi_dataset_loader.py`

```python
class MultiDatasetLoader:
    """Load multiple datasets with unified interface."""
    
    def __init__(self, dataset_names: List[str] = None):
        """
        Initialize loader.
        
        Args:
            dataset_names: List of ['mimic_cxr', 'vqa_rad', 'ms_cxr', 'iu_xray']
                          Default: all 4 datasets
        """
        self.datasets = {}
        for name in dataset_names or ['mimic_cxr', 'vqa_rad', 'ms_cxr', 'iu_xray']:
            self.datasets[name] = self._load_dataset(name)
    
    def get_dataset(self, name: str, task: str = 'report_generation') -> Dataset:
        """Get specific dataset filtered by task."""
        # Filter to samples supporting `task`
        # Return unified interface
    
    def get_all_samples(self) -> Dict[str, List[Sample]]:
        """Get samples from all datasets."""
    
    def get_statistics(self) -> Dict:
        """Dataset statistics: count, avg image size, task breakdown."""
```

### Phase 2: Unified Experiment Runner (Week 2, 1.5 hours)

**File:** `experiments/run_multi_dataset_comparison.py`

```python
class MultiDatasetComparison:
    """Run models across multiple datasets."""
    
    def __init__(self, 
                 model_names: List[str] = None,  # 14 MAIN if None
                 dataset_names: List[str] = None,  # all 4 if None
                 output_dir: str = 'results/multi_dataset'):
        self.models = model_names or get_main_benchmark_models()
        self.datasets = MultiDatasetLoader(dataset_names)
        self.results = {}
    
    def run(self, num_samples_per_dataset: int = 200) -> Dict:
        """Run comparison across all models and datasets."""
        progress = tqdm(total=len(self.models) * len(self.datasets.datasets))
        
        for model_name in self.models:
            model = load_model(model_name)
            
            for dataset_name, dataset in self.datasets.datasets.items():
                samples = dataset.get_samples(n=num_samples_per_dataset)
                
                predictions = self._run_model_on_dataset(
                    model, dataset_name, samples
                )
                
                metrics = self._evaluate_predictions(
                    dataset_name, predictions
                )
                
                self.results[(model_name, dataset_name)] = metrics
                progress.update(1)
        
        return self.results
    
    def aggregate_results(self) -> pd.DataFrame:
        """Create cross-dataset comparison table."""
        # Pivot: Rows=Models, Cols=Datasets, Values=BLEU/Accuracy
        # Include aggregation: mean, std, rank
```

### Phase 3: Results Aggregation (Week 2, 1 hour)

**File:** `src/utils/multi_dataset_aggregator.py`

```python
class MultiDatasetAggregator:
    """Aggregate and analyze results across datasets."""
    
    def __init__(self, results_dict: Dict):
        """Load results from multi_dataset_comparison."""
        self.results = results_dict
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create table for papers:
        
        Model          | MIMIC-CXR | VQA-RAD | MS-CXR | IU X-Ray | MEAN
        --------       | --------- | ------- | ------ | -------- | -----
        Qwen2-VL-2B    |   45.2    |  62.1   |  42.1  |   38.5   | 47.0
        LLaVA-Med      |   46.1    |  63.5   |  43.2  |   39.1   | 48.0
        ...
        """
    
    def dataset_performance_profile(self) -> Dict:
        """
        Analyze which datasets are "harder" for models:
        {
            'easy_datasets': ['iu_xray'],  # Models score high
            'hard_datasets': ['ms_cxr'],   # Models score low
            'variance': {...}              # How consistent across datasets
        }
        """
    
    def model_consistency(self) -> Dict:
        """
        Rank models by consistency across datasets:
        {
            'consistent_models': ['llava-med', ...],
            'variable_models': [...]
        }
        """
```

### Phase 4: Statistical Analysis (Week 2, 1.5 hours)

**File:** `src/utils/multi_dataset_stats.py`

```python
class MultiDatasetStatisticalTester:
    """Statistical testing across datasets."""
    
    def anova_per_dataset(self) -> Dict:
        """ANOVA: Are models significantly different within each dataset?"""
        # For each dataset, run ANOVA on model scores
        # Return: F-statistic, p-value, effect_size
    
    def model_x_dataset_interaction(self) -> Dict:
        """2-way ANOVA: Model × Dataset interaction."""
        # Q: Do models have consistent ranking across datasets?
        # Q: Do some datasets favor certain models?
    
    def consistency_ranking(self) -> List[str]:
        """Rank models by consistency across datasets."""
        # Models with low variance across datasets rank high
    
    def generalization_score(self) -> Dict:
        """Calculate "generalization" for each model."""
        # High generalization = performs well on all datasets
        # Low generalization = dataset-specific performance
```

---

## File Structure

```
results/
└── multi_dataset/
    ├── run_20250110_150000/          # Timestamped run
    │   ├── config.yaml               # Run configuration
    │   ├── raw_predictions/
    │   │   ├── qwen2-vl-2b_mimic_cxr.json
    │   │   ├── qwen2-vl-2b_vqa_rad.json
    │   │   ├── llava-med_mimic_cxr.json
    │   │   └── ...
    │   ├── metrics/
    │   │   ├── per_model_per_dataset_metrics.json
    │   │   ├── aggregate_comparison_table.csv
    │   │   └── statistical_tests.json
    │   ├── analysis/
    │   │   ├── dataset_difficulty_ranking.md
    │   │   ├── model_generalization_scores.md
    │   │   └── cross_dataset_plots.png
    │   └── report.md                 # Executive summary
```

---

## Usage Examples

### Example 1: Basic Multi-Dataset Run

```python
from experiments.run_multi_dataset_comparison import MultiDatasetComparison

# Run all 14 MAIN models on all 4 datasets
comparison = MultiDatasetComparison()
results = comparison.run(num_samples_per_dataset=200)

# Get comparison table
table = comparison.aggregate_results()
print(table)
```

### Example 2: Specific Models + Datasets

```python
# Test only 3 models on 2 datasets
comparison = MultiDatasetComparison(
    model_names=['qwen2-vl-2b', 'llava-med', 'chexagent'],
    dataset_names=['mimic_cxr', 'vqa_rad']
)

results = comparison.run(num_samples_per_dataset=100)
```

### Example 3: Statistical Analysis

```python
from src.utils.multi_dataset_stats import MultiDatasetStatisticalTester

stats = MultiDatasetStatisticalTester(results)

# Which models are most consistent across datasets?
consistent = stats.consistency_ranking()
print(f"Most consistent: {consistent[0]}")

# Is there model × dataset interaction?
interaction = stats.model_x_dataset_interaction()
print(f"Interaction p-value: {interaction['p_value']}")
```

### Example 4: Full Notebook Workflow

```python
# notebooks/multi_dataset_analysis.ipynb
from experiments.run_multi_dataset_comparison import MultiDatasetComparison
from src.utils.multi_dataset_aggregator import MultiDatasetAggregator

# Step 1: Run comparison
comparison = MultiDatasetComparison()
raw_results = comparison.run()

# Step 2: Aggregate
aggregator = MultiDatasetAggregator(raw_results)
comparison_table = aggregator.create_comparison_table()

# Step 3: Visualize
import matplotlib.pyplot as plt
aggregator.plot_heatmap()  # Model × Dataset heatmap
aggregator.plot_box_plots()  # Distribution per dataset

# Step 4: Analyze
consistency = aggregator.model_consistency()
generalization = aggregator.generalization_score()
```

---

## Key Metrics & Outputs

### Per-Model-Per-Dataset Metrics
```json
{
  "qwen2-vl-2b": {
    "mimic_cxr": {
      "bleu": 45.2,
      "rouge_l": 0.38,
      "radgraph_f1": 0.52,
      "hallucination_rate": 0.08,
      "vqa_accuracy": null
    },
    "vqa_rad": {
      "bleu": null,
      "vqa_accuracy": 0.62,
      "hallucination_rate": 0.06
    }
  }
}
```

### Aggregation Output (CSV)
```
Model,MIMIC-CXR (BLEU),VQA-RAD (Acc),MS-CXR (BLEU),IU X-Ray (BLEU),Mean,Std,Rank
qwen2-vl-2b,45.2,0.62,42.1,38.5,42.3,2.8,3
llava-med,46.1,0.63,43.2,39.1,43.0,2.9,2
llama3-vision,47.8,0.61,44.5,40.2,43.8,3.4,1
...
```

### Statistical Tests
```json
{
  "anova_per_dataset": {
    "mimic_cxr": { "f_statistic": 12.5, "p_value": 0.0001, "effect_size": 0.45 },
    "vqa_rad": { "f_statistic": 8.3, "p_value": 0.0005, "effect_size": 0.32 }
  },
  "model_x_dataset_interaction": {
    "f_statistic": 2.1,
    "p_value": 0.018,
    "interpretation": "Significant interaction - models don't rank consistently"
  },
  "most_consistent_models": ["llava-med", "qwen2-vl-2b", "phi3-vision"]
}
```

---

## Implementation Roadmap

| Phase | Task | Effort | Status |
|-------|------|--------|--------|
| 1 | Create `MultiDatasetLoader` | 1 h | ⏳ Week 2 |
| 2 | Build `MultiDatasetComparison` runner | 1.5 h | ⏳ Week 2 |
| 3 | Implement `MultiDatasetAggregator` | 1 h | ⏳ Week 2 |
| 4 | Add statistical testing (`MultiDatasetStatisticalTester`) | 1.5 h | ⏳ Week 2 |
| 5 | Create visualization helper | 0.5 h | ⏳ Week 2 |
| 6 | Write `notebooks/multi_dataset_analysis.ipynb` | 2 h | ⏳ Week 3 |
| **Total** | **Multi-Dataset Support** | **7.5 h** | **Planned** |

---

## Integration Points

### With Existing Code

1. **Model Loading:** Uses existing `load_model()` from model_registry
2. **Datasets:** Extends existing `BaseDataset` for unified interface
3. **Evaluation:** Reuses existing evaluators (BLEU, RadGraph, VQA)
4. **Results:** Compatible with existing ResultWriter format

### Command-Line Interface

```bash
# Run multi-dataset comparison
python experiments/run_multi_dataset_comparison.py \
    --models main \
    --datasets mimic_cxr,vqa_rad,ms_cxr,iu_xray \
    --num-samples 200 \
    --output results/multi_dataset_20250110

# Or with subset
python experiments/run_multi_dataset_comparison.py \
    --models qwen2-vl-2b llava-med chexagent \
    --datasets mimic_cxr vqa_rad \
    --num-samples 100
```

---

## Benefits & Insights

### For Research
- **Generalization:** Which models generalize across datasets?
- **Dataset Bias:** Are models dataset-specific?
- **Robustness:** Model performance under different conditions
- **Statistical Power:** Larger sample pool across datasets

### For Practitioners
- **Model Selection:** Choose model best for your dataset combination
- **Transfer Learning:** Pre-training dataset effects
- **Domain Adaptation:** Dataset-specific tuning needs

### For Papers
- Comprehensive comparison tables
- Multi-dataset ablation studies
- Cross-dataset consistency ranking
- Generalization metrics for model cards

---

## Future Enhancements

1. **Streaming Pipeline:** Process large datasets in batches
2. **Distributed Execution:** Run on multiple GPUs in parallel
3. **Caching:** Cache predictions to avoid re-running models
4. **Differential Analysis:** Compare model pairs across datasets
5. **Error Analysis:** Common failure patterns per dataset
6. **Human Evaluation:** Benchmark against radiologist performance

---

## Notes

- **Runtime:** Full run (14 models × 4 datasets × 200 samples) ≈ 6-8 hours on single T4 GPU
- **Storage:** Raw predictions + metrics ≈ 5-10 GB depending on sampling
- **Dependencies:** No new packages required (uses existing)
- **Backward Compatible:** Existing single-dataset runs still work unchanged

---

**Last Updated:** Week 1-2 Architecture Design  
**Next Steps:** Implement Phase 1 (MultiDatasetLoader) in Week 2
