"""
Method comparison bar charts with significance testing.

Compares multiple methods with error bars and statistical significance brackets.
"""

from __future__ import annotations

from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from src.visualize.style import (
    nature_style,
    save_figure,
    SINGLE_COL,
    annotate_significance,
    compute_statistical_test,
    create_nature_palette,
)


def plot_method_comparison(
    results_dict: Dict[str, Dict[str, float]],
    out_path: Optional[str] = None,
    metric_name: str = "Score",
    test: str = "ttest",
    baseline: Optional[str] = None,
) -> None:
    """
    Plot bar chart comparing multiple methods with significance.

    Args:
        results_dict: Dict of {method: {mean: float, std: float, samples: List[float]}}
        out_path: Output path for figure (optional)
        metric_name: Name of metric for y-axis
        test: Statistical test to use ('ttest', 'mannwhitney', 'wilcoxon')
        baseline: Method to compare against (optional)
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    means = [results_dict[m]["mean"] for m in methods]
    stds = [results_dict[m]["std"] for m in methods]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.6))

    with nature_style():
        palette = create_nature_palette(n_methods)
        x_pos = np.arange(n_methods)

        ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=3,
            color=palette,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 1},
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        if baseline and baseline in results_dict:
            baseline_idx = methods.index(baseline)
            baseline_samples = results_dict[baseline].get("samples", [])

            for i, method in enumerate(methods):
                if method == baseline:
                    continue

                method_samples = results_dict[method].get("samples", [])
                if len(baseline_samples) > 0 and len(method_samples) > 0:
                    _, pvalue = compute_statistical_test(
                        np.array(baseline_samples), np.array(method_samples), test
                    )

                    y_max = max(means[baseline_idx], means[i]) + max(
                        stds[baseline_idx], stds[i]
                    )
                    annotate_significance(ax, baseline_idx, i, y_max * 1.05, pvalue)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()
