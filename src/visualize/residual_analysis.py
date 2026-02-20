"""
Residual analysis visualization.

Analyzes spatial patterns in prediction errors.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.visualize.style import (
    nature_style,
    save_figure,
    SINGLE_COL,
    DOUBLE_COL,
    add_colorbar,
    CMAP_ERROR_SIGNED,
)


def plot_residual_analysis(
    x: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot spatial residual patterns.

    Args:
        x: X coordinates (2D mesh)
        y: Y coordinates (2D mesh)
        residuals: Residual values (predicted - true)
        out_path: Output path for figure (optional)
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    with nature_style():
        vmax = np.percentile(np.abs(residuals), 95)

        im = ax.pcolormesh(
            x,
            y,
            residuals,
            cmap=CMAP_ERROR_SIGNED,
            shading="auto",
            vmax=vmax,
            vmin=-vmax,
        )

        add_colorbar(ax, im, label="Residual")

        ax.set_xlabel("X (km)", fontsize=9)
        ax.set_ylabel("Y (km)", fontsize=9)

        if title:
            ax.set_title(title, fontsize=9)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()


def plot_residual_components(
    residuals: np.ndarray,
    out_path: Optional[str] = None,
    labels: Optional[list[str]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot residual components as grouped bar chart.

    Args:
        residuals: Array of residual values (N x components)
        out_path: Output path for figure (optional)
        labels: Labels for each component
        title: Plot title (optional)
    """
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)

    n_components = residuals.shape[1]
    if labels is None:
        labels = [f"Component {i + 1}" for i in range(n_components)]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.5))

    with nature_style():
        means = np.mean(residuals, axis=0)
        stds = np.std(residuals, axis=0)

        x_pos = np.arange(n_components)

        ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=3,
            color="#0072B2",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.axhline(0, color="gray", linestyle="--", lw=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Residual", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        if title:
            ax.set_title(title, fontsize=9)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()
