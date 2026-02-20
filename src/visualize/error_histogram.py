"""
Error histogram visualization with KDE overlay.

Displays error distribution with optional kernel density estimation.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.visualize.style import nature_style, save_figure, SINGLE_COL


def plot_error_histogram(
    errors: np.ndarray,
    out_path: Optional[str] = None,
    bins: int = 50,
    kde: bool = True,
    rug: bool = False,
    xlabel: str = "Error",
    title: Optional[str] = None,
) -> None:
    """
    Plot error distribution with optional KDE overlay.

    Args:
        errors: Array of error values
        out_path: Output path for figure (optional)
        bins: Number of histogram bins
        kde: Whether to show KDE curve
        rug: Whether to show rug plot
        xlabel: X-axis label
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.5))

    with nature_style():
        ax.hist(
            errors,
            bins=bins,
            density=True,
            alpha=0.7,
            color="#0072B2",
            edgecolor="white",
        )

        if kde:
            sns.kdeplot(errors, ax=ax, color="#D55E00", lw=1.5, label="KDE")

        if rug:
            sns.rugplot(errors, ax=ax, color="#009E73", alpha=0.3, height=0.05)

        mean_err = np.mean(errors)
        median_err = np.median(errors)
        ax.axvline(
            mean_err,
            color="#D55E00",
            linestyle="--",
            lw=1,
            label=f"Mean: {mean_err:.4f}",
        )
        ax.axvline(
            median_err,
            color="#CC79A7",
            linestyle=":",
            lw=1,
            label=f"Median: {median_err:.4f}",
        )

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)

        if title:
            ax.set_title(title, fontsize=9)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()
