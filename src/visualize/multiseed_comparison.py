"""
Multi-seed comparison with confidence intervals.

Aggregates results across multiple random seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from src.visualize.style import nature_style, save_figure, SINGLE_COL


def plot_multiseed_comparison(
    run_dirs: List[str],
    out_path: Optional[str] = None,
    metric: str = "loss",
    title: Optional[str] = None,
) -> None:
    """
    Compare multiple runs with confidence intervals.

    Args:
        run_dirs: List of run directory paths
        out_path: Output path for figure (optional)
        metric: Metric to plot ('loss', 'loss_data', etc.)
        title: Plot title (optional)
    """
    all_histories = []

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        if run_path.is_file():
            with open(run_path, "r") as f:
                all_histories.append(json.load(f))
            continue

        history_file = run_path / "training_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                all_histories.append(json.load(f))
            continue

        results_file = Path("results/tables/training_history.json")
        if results_file.exists():
            with open(results_file, "r") as f:
                all_histories.append(json.load(f))

    if not all_histories or metric not in all_histories[0]:
        print(f"Warning: Metric '{metric}' not found in histories")
        return

    min_len = min(len(h[metric]) for h in all_histories)
    data = np.array([h[metric][:min_len] for h in all_histories])

    epochs = np.arange(1, min_len + 1)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.6))

    with nature_style():
        ax.plot(epochs, mean, color="#0072B2", lw=1.5, label="Mean")
        ax.fill_between(
            epochs, mean - std, mean + std, alpha=0.3, color="#0072B2", label="±1 std"
        )

        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(metric.capitalize(), fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title, fontsize=9)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()


def _load_aggregate_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load aggregate.csv and return dict of column arrays."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    return {col: df[col].values for col in df.columns}
