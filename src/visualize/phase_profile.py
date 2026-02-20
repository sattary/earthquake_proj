"""
Phase profile visualization for strain azimuth.

Domain-specific plot for wrapped phase data (strain azimuth in radians).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.visualize.style import (
    nature_style,
    save_figure,
    SINGLE_COL,
    add_colorbar,
    CMAP_PHASE,
)


def plot_phase_profile(
    x: np.ndarray,
    phase: np.ndarray,
    out_path: Optional[str] = None,
    xlabel: str = "Distance (km)",
    ylabel: str = "Phase (rad)",
    title: Optional[str] = None,
    wrap: bool = True,
) -> None:
    """
    Plot 1D phase profile with optional wrapping.

    Args:
        x: X coordinates
        phase: Phase values in radians
        out_path: Output path for figure (optional)
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title (optional)
        wrap: Whether to wrap phase to [-π, π]
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.4))

    if wrap:
        phase = np.angle(np.exp(1j * phase))

    with nature_style():
        ax.plot(x, phase, lw=1, color="#0072B2")
        ax.fill_between(x, phase, 0, alpha=0.3, color="#0072B2")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

        if wrap:
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

        if title:
            ax.set_title(title, fontsize=9)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()


def plot_phase_2d(
    x: np.ndarray,
    y: np.ndarray,
    phase: np.ndarray,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot 2D phase map with cyclic colormap.

    Args:
        x: X coordinates (2D mesh)
        y: Y coordinates (2D mesh)
        phase: Phase values in radians
        out_path: Output path for figure (optional)
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    with nature_style():
        phase_wrapped = np.angle(np.exp(1j * phase))

        im = ax.pcolormesh(x, y, phase_wrapped, cmap=CMAP_PHASE, shading="auto")

        add_colorbar(ax, im, label="Phase (rad)")

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
