"""
Coulomb Failure Function (CFF) map visualization.

Domain-specific plot for visualizing seismogenic potential.
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
    CMAP_DIVERGING,
)


def plot_cff_map(
    x: np.ndarray,
    y: np.ndarray,
    cff: np.ndarray,
    earthquake_x: Optional[np.ndarray] = None,
    earthquake_y: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
    title: str = "Coulomb Failure Function Map",
    vmax: Optional[float] = None,
) -> None:
    """
    Plot 2D Coulomb Failure Function (CFF) map.

    Args:
        x: X coordinates (2D mesh, km)
        y: Y coordinates (2D mesh, km)
        cff: CFF values (MPa)
        earthquake_x: Optional X coordinates of raw earthquakes to overlay
        earthquake_y: Optional Y coordinates of raw earthquakes to overlay
        out_path: Output path for figure (optional)
        title: Plot title
        vmax: Maximum absolute value for diverging colormap
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    if vmax is None:
        vmax = max(abs(np.percentile(cff, 5)), abs(np.percentile(cff, 95)))

    with nature_style():
        im = ax.pcolormesh(
            x,
            y,
            cff,
            cmap=CMAP_DIVERGING,
            shading="auto",
            vmin=-vmax,
            vmax=vmax,
        )

        add_colorbar(ax, im, label="CFF (MPa)")

        if earthquake_x is not None and earthquake_y is not None:
            ax.scatter(
                earthquake_x,
                earthquake_y,
                s=2,
                c="k",
                alpha=0.5,
                marker=".",
                label="Earthquakes",
            )
            ax.legend(loc="upper right", fontsize=7)

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
