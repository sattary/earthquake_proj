"""
Loss landscape visualization.

2D contour plots showing loss surface around trained parameters.
"""

from __future__ import annotations

from typing import Optional, Callable

import matplotlib.pyplot as plt
import numpy as np

from src.visualize.style import (
    nature_style,
    save_figure,
    SINGLE_COL,
    add_colorbar,
    CMAP_ERROR_ABS,
)


def plot_loss_landscape(
    loss_fn: Callable[[np.ndarray], float],
    param_init: np.ndarray,
    param_opt: np.ndarray,
    out_path: Optional[str] = None,
    grid_size: int = 20,
    scale: float = 0.5,
    title: Optional[str] = None,
) -> None:
    """
    Plot 2D loss landscape contour.

    Args:
        loss_fn: Function that takes 1D param array and returns loss
        param_init: Initial parameter values
        param_opt: Optimized parameter values
        out_path: Output path for figure (optional)
        grid_size: Number of points in each dimension
        scale: Fraction of distance from init to opt to plot
        title: Plot title (optional)
    """
    direction = param_opt - param_init
    distance = np.linalg.norm(direction)
    unit_dir = direction / distance

    perp_dir = np.array([-unit_dir[1], unit_dir[0]])
    if len(perp_dir) != len(param_init):
        perp_dir = np.zeros_like(unit_dir)

    range_val = distance * scale
    t = np.linspace(-range_val, range_val, grid_size)
    s = np.linspace(-range_val, range_val, grid_size)
    T, S = np.meshgrid(t, s)

    losses = np.zeros_like(T)

    for i in range(grid_size):
        for j in range(grid_size):
            params = param_opt + unit_dir * T[i, j] + perp_dir * S[i, j]
            losses[i, j] = loss_fn(params)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    with nature_style():
        levels = np.logspace(np.log10(losses.min()), np.log10(losses.max()), 20)

        im = ax.contourf(T, S, losses, levels=levels, cmap=CMAP_ERROR_ABS)
        ax.contour(
            T, S, losses, levels=levels, colors="white", linewidths=0.3, alpha=0.5
        )

        add_colorbar(ax, im, label="Loss")

        ax.plot(0, 0, "w*", markersize=15, label="Optimum")
        ax.plot(T.max(), S.max(), "wx", markersize=10, label="Initial direction")

        ax.set_xlabel("Direction 1", fontsize=9)
        ax.set_ylabel("Direction 2", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)

        if title:
            ax.set_title(title, fontsize=9)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()
