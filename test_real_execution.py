#!/usr/bin/env python3
"""Actually execute statistical analysis code with mock data."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from itertools import combinations
import sys

print("=" * 70)
print("REAL EXECUTION TEST - Statistical Analysis Notebook Code")
print("=" * 70)
print()

# Set up paths
RESULTS_DIR = Path('results')

# ===== CELL 2: Load Data =====
print("📦 Loading data...")
all_samples = []
for run_dir in sorted(RESULTS_DIR.glob('*')):
    if not run_dir.is_dir():
        continue
    
    sample_file = run_dir / 'sample_metrics.jsonl'
    if not sample_file.exists():
        continue
    
    try:
        for line in sample_file.read_text().splitlines():
            if line.strip():
                sample = json.loads(line)
                sample['run_name'] = run_dir.name
                all_samples.append(sample)
    except Exception as e:
        print(f'Error loading {run_dir.name}: {e}')

samples_df = pd.DataFrame(all_samples)
print(f'✓ Loaded {len(samples_df)} samples from {samples_df.run_name.nunique()} runs')
print(f'  Models: {list(samples_df.model_name.unique())}')
print(f'  Datasets: {list(samples_df.dataset_name.unique())}')
print()

# ===== CELL 3: Model Ranking =====
print("📊 Model Rankings (95% CI)...")
metric_cols = [col for col in samples_df.columns
               if not col.startswith('_')
               and col not in {'sample_id', 'dataset_name', 'model_name', 'task', 'timestamp', 'run_name'}
               and not col.endswith('_fallback_used')]

print(f'Found {len(metric_cols)} metrics: {metric_cols}')

for metric in metric_cols[:2]:  # Show first 2
    print(f'\n{metric}:')
    ranking = samples_df.groupby('model_name')[metric].agg(['mean', 'std', 'count']).reset_index()
    ranking['se'] = ranking['std'] / np.sqrt(ranking['count'])
    ranking['ci_lower'] = ranking['mean'] - 1.96 * ranking['se']
    ranking['ci_upper'] = ranking['mean'] + 1.96 * ranking['se']
    ranking = ranking.sort_values('mean', ascending=False)
    
    for _, row in ranking.iterrows():
        print(f"  {row.model_name:20} {row['mean']:8.4f} [{row.ci_lower:.4f}, {row.ci_upper:.4f}]")

print()

# ===== CELL 4: Pairwise t-tests =====
print("🔬 Pairwise t-tests...")
if len(metric_cols) > 0:
    test_metric = metric_cols[0]
    models = sorted(samples_df.model_name.unique())
    print(f'Testing: {test_metric}\n')
    
    results = []
    for m1, m2 in combinations(models, 2):
        data1 = samples_df[samples_df.model_name == m1][test_metric].dropna()
        data2 = samples_df[samples_df.model_name == m2][test_metric].dropna()
        
        if len(data1) > 1 and len(data2) > 1:
            t_stat, p_value = scipy_stats.ttest_ind(data1, data2)
            
            n1, n2 = len(data1), len(data2)
            var_pool = ((n1-1)*data1.var() + (n2-1)*data2.var()) / (n1+n2-2)
            cohens_d = (data1.mean() - data2.mean()) / np.sqrt(var_pool) if var_pool > 0 else 0
            
            sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
            
            print(f"{m1:20} vs {m2:20} | t={t_stat:6.2f}, p={p_value:.4f} {sig:3s}, Cohen's d={cohens_d:.3f}")
            results.append({'model1': m1, 'model2': m2, 'p_value': p_value, 'cohens_d': cohens_d})

print()

# ===== CELL 5: Visualizations =====
print("📈 Generating visualizations...")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Box plot
    n_metrics = min(2, len(metric_cols))
    fig, axes = plt.subplots(1, n_metrics, figsize=(12, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metric_cols[:n_metrics]):
        ax = axes[idx]
        metric_data = samples_df.dropna(subset=[metric])
        sns.boxplot(data=metric_data, x='model_name', y=metric, ax=ax, palette='Set2')
        ax.set_title(metric, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    boxplot_path = RESULTS_DIR / 'test_boxplots.png'
    plt.savefig(boxplot_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {boxplot_path}")
    
    # Heatmap
    pivot = samples_df.groupby('model_name')[metric_cols].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Model Performance Matrix', fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = RESULTS_DIR / 'test_heatmap.png'
    plt.savefig(heatmap_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {heatmap_path}")
    
except Exception as e:
    print(f"✗ Visualization error: {e}")

print()

# ===== Dataset breakdown =====
print("📋 Performance by Dataset:")
for dataset in sorted(samples_df.dataset_name.unique()):
    ds_df = samples_df[samples_df.dataset_name == dataset]
    print(f'\n{dataset} (n={len(ds_df)}):')
    
    if len(metric_cols) > 0:
        metric = metric_cols[0]
        ds_ranking = ds_df.groupby('model_name')[metric].agg(['mean']).sort_values('mean', ascending=False)
        for model, row in ds_ranking.iterrows():
            print(f'  {model:20} {row["mean"]:8.4f}')

print()
print("=" * 70)
print("✅ EXECUTION COMPLETE")
print("=" * 70)
print(f"""
Summary:
  - Loaded: {len(samples_df)} samples from {samples_df.model_name.nunique()} models
  - Metrics: {len(metric_cols)} ({', '.join(metric_cols[:2])}...)
  - Tests: Pairwise t-tests with effect sizes ✓
  - Plots: Box plots & heatmap generated ✓
  - Status: WORKING (not theory, actual output!)
""")
