"""
Training curve visualization with confidence bands.

Plots loss convergence from training history with optional multi-seed support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.visualize.style import (
    nature_style,
    save_figure,
    SINGLE_COL,
    DOUBLE_COL,
)


def _load_history(run_dir: str) -> dict:
    """Load training history from JSON file."""
    history_path = Path(run_dir)

    if history_path.is_file():
        with open(history_path, "r") as f:
            return json.load(f)

    history_file = history_path / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            return json.load(f)

    results_table = Path("results/tables/training_history.json")
    if results_table.exists():
        with open(results_table, "r") as f:
            return json.load(f)

    raise FileNotFoundError(f"Training history not found in {run_dir}")


def plot_training_curve(
    run_dir: str,
    out_path: Optional[str] = None,
    show_lr: bool = False,
    confidence: float = 0.95,
    width: str = "single",
) -> None:
    """
    Plot training loss and validation metrics with confidence bands.

    Args:
        run_dir: Path to run directory or history file
        out_path: Output path for figure (optional)
        show_lr: Whether to show learning rate subplot
        confidence: Confidence level for bands (0-1)
        width: 'single' or 'double' column width
    """
    history = _load_history(run_dir)

    if not history or "loss" not in history:
        print(f"Warning: No training history found in {run_dir}")
        return

    epochs = np.arange(1, len(history["loss"]) + 1)

    fig_width = SINGLE_COL if width == "single" else DOUBLE_COL
    if show_lr:
        fig, axes = plt.subplots(
            2, 1, figsize=(fig_width, fig_width * 0.8), sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.6))

    with nature_style():
        if show_lr:
            ax = axes[0]

        loss = np.array(history["loss"])
        ax.plot(epochs, loss, label="Total Loss", color="#0072B2", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        if "loss_data" in history:
            loss_data = np.array(history["loss_data"])
            ax.plot(
                epochs, loss_data, label="Data Loss", color="#D55E00", lw=0.8, alpha=0.8
            )

        if "loss_pde" in history:
            loss_pde = np.array(history["loss_pde"])
            ax.plot(
                epochs, loss_pde, label="PDE Loss", color="#009E73", lw=0.8, alpha=0.8
            )

        if show_lr and "lr" in history:
            axes[1].plot(epochs, history["lr"], color="#CC79A7", lw=1)
            axes[1].set_ylabel("Learning Rate", fontsize=9)
            axes[1].set_yscale("log")
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlabel("Epoch", fontsize=9)
        elif not show_lr:
            ax.set_xlabel("Epoch", fontsize=9)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()
