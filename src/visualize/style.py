"""
Nature-Compliant Style System for Publication-Quality Visualizations.

Provides consistent styling, color palettes, and utilities for creating
figures that meet Nature journal standards.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional, Tuple, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from scipy import stats


MM_TO_INCH = 1.0 / 25.4

SINGLE_COL = 89 * MM_TO_INCH
DOUBLE_COL = 183 * MM_TO_INCH
DPI = 300

NATURE_PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "teal": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "reddish_purple": "#CC79A7",
}

CMAP_PHASE = "twilight"
CMAP_INTENSITY = "gray"
CMAP_ERROR_SIGNED = "RdBu_r"
CMAP_ERROR_ABS = "inferno"
CMAP_CONTINUOUS = "cividis"
CMAP_DIVERGING = "coolwarm"

NATURE_RC = {
    "font.family": "serif",
    "font.serif": ["Computer Modern", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


_original_rc = {}


def _store_original_rc():
    """Store original rcParams to restore later."""
    global _original_rc
    if not _original_rc:
        _original_rc = {k: mpl.rcParams.get(k) for k in NATURE_RC.keys()}


@contextmanager
def nature_style() -> Generator[None, None, None]:
    """
    Context manager that temporarily applies Nature publication style.

    Usage:
        with nature_style():
            fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.6))
            ax.plot(x, y)
            save_figure(fig, "output")

    Yields:
        None - applies settings to matplotlib globally within context
    """
    _store_original_rc()
    try:
        mpl.rcParams.update(NATURE_RC)
        sns.set_theme(style="whitegrid", context="paper")
        yield
    finally:
        for key, value in _original_rc.items():
            if value is not None:
                mpl.rcParams[key] = value


def save_figure(
    fig: Figure,
    path: str,
    formats: Optional[list[str]] = None,
    dpi: int = 600,
) -> None:
    """
    Save figure in multiple formats (PNG, PDF, SVG).

    Args:
        fig: Matplotlib figure to save
        path: Output path without extension
        formats: List of formats to save (default: ['png', 'pdf', 'svg'])
        dpi: DPI for raster formats (PNG)
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]

    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == "png":
            fig.savefig(
                p.with_suffix(".png"),
                bbox_inches="tight",
                dpi=dpi,
                transparent=False,
            )
        else:
            fig.savefig(
                p.with_suffix(f".{fmt}"),
                bbox_inches="tight",
                transparent=False,
            )

    plt.close(fig)
    print(f"Exported figure: {p.stem}.{{{','.join(formats)}}} to {p.parent}")


def add_colorbar(
    ax: Any,
    im: Any,
    label: Optional[str] = None,
    orientation: str = "vertical",
    shrink: float = 0.8,
    **kwargs,
) -> Colorbar:
    """
    Add a properly sized colorbar to an axis.

    Args:
        ax: Matplotlib axis
        im: Image (result of ax.imshow or ax.pcolormesh)
        label: Colorbar label
        orientation: 'vertical' or 'horizontal'
        shrink: Shrink factor for colorbar size

    Returns:
        Colorbar object
    """
    cbar = plt.colorbar(im, ax=ax, orientation=orientation, shrink=shrink, **kwargs)

    if label:
        cbar.set_label(label, fontsize=9)

    cbar.ax.tick_params(labelsize=7)

    return cbar


def annotate_significance(
    ax: Any,
    x1: float,
    x2: float,
    y: float,
    pvalue: float,
    h: float = 0.02,
    text_offset: float = 0.01,
) -> None:
    """
    Draw a significance bracket with stars.

    Args:
        ax: Matplotlib axis
        x1: Left x position
        x2: Right x position
        y: Y position for bracket
        pvalue: P-value to display
        h: Bracket height
        text_offset: Vertical offset for text
    """
    stars = format_pvalue(pvalue)

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c="black")
    ax.text(
        (x1 + x2) / 2, y + h + text_offset, stars, ha="center", va="bottom", fontsize=9
    )


def compute_statistical_test(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = "ttest",
) -> Tuple[float, float]:
    """
    Compute statistical test between two groups.

    Args:
        group1: First group of values
        group2: Second group of values
        test: Test type - 'ttest', 'mannwhitney', or 'wilcoxon'

    Returns:
        Tuple of (statistic, pvalue)
    """
    if test == "ttest":
        statistic, pvalue = stats.ttest_ind(group1, group2)
    elif test == "mannwhitney":
        statistic, pvalue = stats.mannwhitneyu(group1, group2)
    elif test == "wilcoxon":
        statistic, pvalue = stats.wilcoxon(group1, group2)
    else:
        raise ValueError(
            f"Unknown test: {test}. Use 'ttest', 'mannwhitney', or 'wilcoxon'"
        )

    return statistic, pvalue


def format_pvalue(pval: float) -> str:
    """
    Format p-value for display with significance stars.

    Args:
        pval: P-value

    Returns:
        Formatted string like '***', '**', '*', or 'ns'
    """
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return "ns"


def create_nature_palette(n_colors: int = 6) -> list[str]:
    """
    Generate Nature-compliant color list.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of hex color codes
    """
    palette_values = list(NATURE_PALETTE.values())

    if n_colors <= len(palette_values):
        return palette_values[:n_colors]

    return sns.color_palette("husl", n_colors).as_hex()


def setup_figure(
    width: str = "single",
    aspect_ratio: float = 0.6,
    style: bool = True,
) -> Tuple[Figure, Any]:
    """
    Create a properly sized figure.

    Args:
        width: 'single' (89mm) or 'double' (183mm)
        as aspect_ratio: Height fraction of width
        style: Whether to apply nature_style

    Returns:
        Tuple of (figure, axis)
    """
    if width == "single":
        fig_width = SINGLE_COL
    elif width == "double":
        fig_width = DOUBLE_COL
    else:
        fig_width = float(width)

    fig_height = fig_width * aspect_ratio

    if style:
        with nature_style():
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    return fig, ax
