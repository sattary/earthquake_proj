"""
Prediction scatter plot with identity line.

Displays predicted vs ground truth with R² annotation.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.visualize.style import nature_style, save_figure, SINGLE_COL


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Optional[str] = None,
    max_samples: int = 1000,
    xlabel: str = "Ground Truth",
    ylabel: str = "Prediction",
    title: Optional[str] = None,
    hexbin: bool = False,
) -> None:
    """
    Plot predicted vs ground truth scatter with identity line.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        out_path: Output path for figure (optional)
        max_samples: Maximum samples to plot (for large datasets)
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title (optional)
        hexbin: Use hexbin for large datasets
    """
    if len(y_true) > max_samples:
        idx = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))

    with nature_style():
        if hexbin and len(y_true) > 500:
            hb = ax.hexbin(y_true, y_pred, gridsize=30, cmap="Blues", mincnt=1)
            plt.colorbar(hb, ax=ax, label="Count")
        else:
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, c="#0072B2", edgecolor="none")

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            "k--",
            lw=1,
            label="Identity",
        )

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        textstr = f"$R^2$ = {r2:.3f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_aspect("equal", adjustable="box")

        if title:
            ax.set_title(title, fontsize=9)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()
