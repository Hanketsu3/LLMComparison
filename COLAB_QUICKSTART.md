# 🚀 Colab Quick Start - Statistical Analysis Notebook

## TL;DR
```
1. Upload project to Google Drive
2. Open: notebooks/97_statistical_analysis.ipynb in Colab
3. Run cells top → bottom
4. Done! Visualizations auto-display.
```

## Full Setup (2 minutes)

### Step 1: Prepare Project
```bash
# Option A: From GitHub
git clone https://github.com/your-org/LLMComparison.git

# Option B: From local backup
# Upload LLMComparison folder to Google Drive
```

### Step 2: Mount & Navigate in Colab
```python
# Cell 0 (add if needed):
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/LLMComparison  # adjust path to your upload location
```

### Step 3: Run Notebook
```
Open: notebooks/97_statistical_analysis.ipynb
Execute cells: [Cell 1] → [Cell 2] → ... → [Cell 9]
```

### Step 4: View Results
- Inline visualizations in Colab cells
- Saved files: `results/comparison_boxplots.png`, `results/performance_heatmap.png`

---

## What You Get

✨ **Statistical Comparisons**
- Model rankings with 95% confidence intervals
- Pairwise t-tests showing statistical significance (p-values)
- Cohen's d effect sizes

📊 **Visualizations**
- Box plots: Model score distributions
- Heatmap: Model × Metric performance matrix
- All saved as high-quality PNGs (300 DPI)

📈 **Analysis Levels**
- Overall rankings
- Per-model comparisons
- Dataset-specific performance
- Error categorization

---

## Requirements

### Data
- `results/` directory with run outputs
- `sample_metrics.jsonl` files from each run
- (Pre-populated from your local runs)

### Environment
- ✅ Pre-installed in Colab: pandas, numpy, scipy, matplotlib, seaborn
- Nothing to install!

---

## Execution Time
- ~30 seconds: Load data + compute statistics
- ~30 seconds: Generate visualizations
- **Total: ~1 minute**

---

## Example Output

```
Loaded 150 samples from 3 models

=== Model Rankings (95% CI) ===

metric_1:
  model_a              0.8523 [0.8102, 0.8944]
  model_b              0.7821 [0.7234, 0.8408]
  model_c              0.6954 [0.6341, 0.7567]

=== Pairwise Comparisons ===
model_a vs model_b: t=2.34, p=0.021 *, Cohen's d=0.42
model_a vs model_c: t=4.12, p<0.001 ***, Cohen's d=0.89
model_b vs model_c: t=1.89, p=0.063 ns, Cohen's d=0.31

✓ Saved: comparison_boxplots.png
✓ Saved: performance_heatmap.png
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: results/` | Ensure results/ directory exists with outputs from runs |
| Empty DataFrames | Check that sample_metrics.jsonl files exist in results/ |
| Import errors | Restart runtime: Runtime → Restart Session |
| Visualization not showing | Try: `plt.show()` after cell execution |

---

## Next Steps

1. **Explore locally first:**
   ```bash
   python test_statistical_notebook.py
   python colab_compatibility_check.py
   ```

2. **Deploy to Colab**

3. **Generate publication figures** from results/

4. **Share results** with team

---

## Full Documentation

See also:
- `PHASE_7_COMPLETION.md` - Technical details
- `EXECUTION_GUIDE.md` - All guardrail options
- `README.md` - Full system overview

Good luck on Colab! 🚀
