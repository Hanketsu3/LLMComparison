#!/usr/bin/env python3
"""Generate publication-ready statistical analysis notebook."""

import json
import uuid
from pathlib import Path

def gen_id():
    """Generate cell ID."""
    return f"#VSC-{uuid.uuid4().hex[:12]}"

def create_statistical_notebook():
    """Create 97_statistical_analysis.ipynb."""
    
    cells = [
        # Title
        {
            "cell_type": "markdown",
            "id": gen_id(),
            "metadata": {"language": "markdown"},
            "source": [
                "# 97: Statistical Analysis & Model Comparison\n",
                "\n",
                "Publication-ready statistical framework:\n",
                "- Model ranking with 95% CI bands\n",
                "- Pairwise t-tests + effect sizes (Cohen's d)\n",
                "- Performance heatmaps & box plots\n",
                "- Dataset-specific rankings\n",
                "- Error analysis by task"
            ]
        },
        # Imports
        {
            "cell_type": "code",
            "id": gen_id(),
            "metadata": {"language": "python"},
            "source": [
                "import json\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "from scipy import stats as scipy_stats\n",
                "from itertools import combinations\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (14, 8)\n",
                "%matplotlib inline\n",
                "\n",
                "RESULTS_DIR = Path('results')\n",
                "print(f'Results directory: {RESULTS_DIR}')"
            ]
        },
        # Load Data
        {
            "cell_type": "code",
            "id": gen_id(),
            "metadata": {"language": "python"},
            "source": [
                "# Load all sample metrics\n",
                "all_samples = []\n",
                "for run_dir in sorted(RESULTS_DIR.glob('*')):\n",
                "    if not run_dir.is_dir():\n",
                "        continue\n",
                "    sample_file = run_dir / 'sample_metrics.jsonl'\n",
                "    if not sample_file.exists():\n",
                "        continue\n",
                "    try:\n",
                "        for line in sample_file.read_text().splitlines():\n",
                "            if line.strip():\n",
                "                sample = json.loads(line)\n",
                "                sample['run_name'] = run_dir.name\n",
                "                all_samples.append(sample)\n",
                "    except Exception as e:\n",
                "        print(f'Error loading {run_dir.name}: {e}')\n",
                "\n",
                "samples_df = pd.DataFrame(all_samples)\n",
                "print(f'Loaded {len(samples_df)} samples')\n",
                "print(f'Models: {samples_df.model_name.nunique()}')\n",
                "print(f'Datasets: {samples_df.dataset_name.nunique()}')\n",
                "display(samples_df.head())"
            ]
        },
        # Model Ranking with CI
        {
            "cell_type": "code",
            "id": gen_id(),
            "metadata": {"language": "python"},
            "source": [
                "# Identify metrics\n",
                "metric_cols = [col for col in samples_df.columns\n",
                "               if not col.startswith('_')\n",
                "               and col not in {'sample_id', 'dataset_name', 'model_name', 'task', 'timestamp', 'run_name'}\n",
                "               and not col.endswith('_fallback_used')]\n",
                "\n",
                "print(f'=== Model Rankings (95% CI) ===')\n",
                "for metric in metric_cols[:3]:\n",
                "    ranking = samples_df.groupby('model_name')[metric].agg(['mean', 'std', 'count']).reset_index()\n",
                "    ranking['se'] = ranking['std'] / np.sqrt(ranking['count'])\n",
                "    ranking['ci_lower'] = ranking['mean'] - 1.96 * ranking['se']\n",
                "    ranking['ci_upper'] = ranking['mean'] + 1.96 * ranking['se']\n",
                "    ranking = ranking.sort_values('mean', ascending=False)\n",
                "    print(f'\\n{metric}:')\n",
                "    for _, row in ranking.iterrows():\n",
                "        print(f\"  {row.model_name:20} {row['mean']:8.4f} [{row.ci_lower:.4f}, {row.ci_upper:.4f}]\")"
            ]
        },
        # Pairwise t-tests
        {
            "cell_type": "code",
            "id": gen_id(),
            "metadata": {"language": "python"},
            "source": [
                "# Pairwise comparisons\n",
                "if len(metric_cols) > 0:\n",
                "    test_metric = metric_cols[0]\n",
                "    models = sorted(samples_df.model_name.unique())\n",
                "    print(f'=== Pairwise t-tests ({test_metric}) ===')\n",
                "    print('* p<0.05, ** p<0.01, *** p<0.001\\n')\n",
                "    \n",
                "    results = []\n",
                "    for m1, m2 in combinations(models, 2):\n",
                "        data1 = samples_df[samples_df.model_name == m1][test_metric].dropna()\n",
                "        data2 = samples_df[samples_df.model_name == m2][test_metric].dropna()\n",
                "        if len(data1) > 1 and len(data2) > 1:\n",
                "            t_stat, p_value = scipy_stats.ttest_ind(data1, data2)\n",
                "            n1, n2 = len(data1), len(data2)\n",
                "            var_pool = ((n1-1)*data1.var() + (n2-1)*data2.var()) / (n1+n2-2)\n",
                "            cohens_d = (data1.mean() - data2.mean()) / np.sqrt(var_pool) if var_pool > 0 else 0\n",
                "            sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))\n",
                "            results.append({'model1': m1, 'model2': m2, 'mean_diff': data1.mean()-data2.mean(), 'cohens_d': cohens_d, 'p_value': p_value, 'sig': sig})\n",
                "    comp_df = pd.DataFrame(results).sort_values('p_value')\n",
                "    display(comp_df.head(10))"
            ]
        },
        # Box plots
        {
            "cell_type": "code",
            "id": gen_id(),
            "metadata": {"language": "python"},
            "source": [
                "# Box plots\n",
                "n_metrics = min(4, len(metric_cols))\n",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                "for idx, metric in enumerate(metric_cols[:n_metrics]):\n",
                "    ax = axes.flat[idx]\n",
                "    metric_data = samples_df.dropna(subset=[metric])\n",
                "    sns.boxplot(data=metric_data, x='model_name', y=metric, ax=ax, palette='Set2')\n",
                "    ax.set_title(metric, fontweight='bold')\n",
                "    ax.tick_params(axis='x', rotation=45)\n",
                "plt.tight_layout()\n",
                "plt.savefig(RESULTS_DIR / 'comparison_boxplots.png', dpi=300, bbox_inches='tight')\n",
                "plt.show()\n",
                "print('✓ Saved: comparison_boxplots.png')"
            ]
        },
        # Heatmap
        {
            "cell_type": "code",
            "id": gen_id(),
            "metadata": {"language": "python"},
            "source": [
                "# Performance heatmap\n",
                "pivot = samples_df.groupby('model_name')[metric_cols].mean()\n",
                "if not pivot.empty:\n",
                "    fig, ax = plt.subplots(figsize=(14, 6))\n",
                "    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Score'}, linewidths=0.5)\n",
                "    ax.set_title('Model Performance Matrix', fontweight='bold', fontsize=13)\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(RESULTS_DIR / 'performance_heatmap.png', dpi=300, bbox_inches='tight')\n",
                "    plt.show()\n",
                "    print('✓ Saved: performance_heatmap.png')"
            ]
        },
        # Dataset breakdown
        {
            "cell_type": "code",
            "id": gen_id(),
            "metadata": {"language": "python"},
            "source": [
                "# Per-dataset rankings\n",
                "print('=== Performance by Dataset ===')\n",
                "for dataset in sorted(samples_df.dataset_name.unique()):\n",
                "    ds_df = samples_df[samples_df.dataset_name == dataset]\n",
                "    print(f'\\n{dataset} (n={len(ds_df)} samples):')\n",
                "    if len(metric_cols) > 0:\n",
                "        metric = metric_cols[0]\n",
                "        ds_ranking = ds_df.groupby('model_name')[metric].agg(['mean', 'count']).sort_values('mean', ascending=False)\n",
                "        for model, (mean_val, count) in ds_ranking.iterrows():\n",
                "            print(f'  {model:20} {mean_val:8.4f} (n={int(count)})')"
            ]
        },
        # Summary
        {
            "cell_type": "code",
            "id": gen_id(),
            "metadata": {"language": "python"},
            "source": [
                "print('\\n=== ANALYSIS COMPLETE ===')\n",
                "print(f'Total samples: {len(samples_df)}')\n",
                "print(f'Models: {samples_df.model_name.nunique()}')\n",
                "print(f'Datasets: {samples_df.dataset_name.nunique()}')\n",
                "print(f'Metrics: {len(metric_cols)}')\n",
                "print('\\nOutputs saved to results/')\n",
                "print('  - comparison_boxplots.png')\n",
                "print('  - performance_heatmap.png')"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    output_path = Path("notebooks") / "97_statistical_analysis.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2))
    print(f"✓ Created {output_path}")

if __name__ == "__main__":
    create_statistical_notebook()
