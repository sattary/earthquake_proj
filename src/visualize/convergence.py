"""
Convergence rate analysis.

Analyzes training convergence speed and stability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.visualize.style import nature_style, save_figure, SINGLE_COL


def plot_convergence(
    run_dir: str,
    out_path: Optional[str] = None,
    metric: str = "loss",
    window: int = 10,
    target: Optional[float] = None,
) -> None:
    """
    Analyze convergence rate and stability.

    Args:
        run_dir: Path to run directory or history file
        out_path: Output path for figure (optional)
        metric: Metric to analyze
        window: Smoothing window size
        target: Target value for convergence (optional)
    """
    history = _load_history(run_dir)

    if metric not in history:
        print(f"Warning: Metric '{metric}' not found")
        return

    values = np.array(history[metric])
    epochs = np.arange(1, len(values) + 1)

    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    smoothed_epochs = np.arange(window // 2, len(values) - window // 2)

    conv_epoch, stable_epoch = _find_convergence(smoothed, smoothed_epochs, window)

    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL, SINGLE_COL * 0.4))

    with nature_style():
        axes[0].plot(epochs, values, alpha=0.3, lw=0.5, color="#0072B2", label="Raw")
        axes[0].plot(
            smoothed_epochs,
            smoothed,
            lw=1,
            color="#D55E00",
            label=f"Smoothed (w={window})",
        )

        if target:
            axes[0].axhline(
                target, color="gray", linestyle="--", lw=1, label=f"Target: {target}"
            )

        if conv_epoch:
            axes[0].axvline(
                conv_epoch,
                color="#009E73",
                linestyle=":",
                lw=1,
                label=f"Convergence: {conv_epoch}",
            )

        axes[0].set_xlabel("Epoch", fontsize=9)
        axes[0].set_ylabel(metric.capitalize(), fontsize=9)
        axes[0].legend(loc="upper right", fontsize=7)
        axes[0].grid(True, alpha=0.3)

        diff = np.abs(np.diff(smoothed))
        axes[1].semilogy(
            smoothed_epochs[1:], diff[1:], alpha=0.5, lw=0.5, color="#0072B2"
        )
        axes[1].set_xlabel("Epoch", fontsize=9)
        axes[1].set_ylabel(f"Δ {metric} (log)", fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()


def _load_history(run_dir: str) -> dict:
    """Load training history."""
    history_path = Path(run_dir)

    if history_path.is_file():
        with open(history_path, "r") as f:
            return json.load(f)

    history_file = history_path / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            return json.load(f)

    results_file = Path("results/tables/training_history.json")
    if results_file.exists():
        with open(results_file, "r") as f:
            return json.load(f)

    raise FileNotFoundError(f"Training history not found in {run_dir}")


def _find_convergence(
    values: np.ndarray, epochs: np.ndarray, window: int
) -> Tuple[Optional[int], Optional[int]]:
    """Find convergence epoch."""
    if len(values) < 2 * window:
        return None, None

    threshold = np.std(values[-window:]) * 2

    for i in range(window, len(values) - window):
        local_std = np.std(values[i : i + window])
        if local_std < threshold:
            return epochs[i], None

    return None, None
