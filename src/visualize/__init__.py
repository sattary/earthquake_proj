"""
Visualization Suite for Earthquake PINN Project.

Publication-quality plotting functions following Nature journal standards.
"""

from src.visualize.style import (
    nature_style,
    save_figure,
    add_colorbar,
    annotate_significance,
    compute_statistical_test,
    format_pvalue,
    create_nature_palette,
    setup_figure,
    SINGLE_COL,
    DOUBLE_COL,
    DPI,
    MM_TO_INCH,
    NATURE_PALETTE,
    CMAP_PHASE,
    CMAP_INTENSITY,
    CMAP_ERROR_SIGNED,
    CMAP_ERROR_ABS,
    CMAP_CONTINUOUS,
    CMAP_DIVERGING,
)

from src.visualize.training_curve import plot_training_curve
from src.visualize.error_histogram import plot_error_histogram
from src.visualize.prediction_scatter import plot_prediction_scatter
from src.visualize.method_comparison import plot_method_comparison
from src.visualize.multiseed_comparison import plot_multiseed_comparison
from src.visualize.convergence import plot_convergence
from src.visualize.phase_profile import plot_phase_profile, plot_phase_2d
from src.visualize.loss_landscape import plot_loss_landscape
from src.visualize.residual_analysis import (
    plot_residual_analysis,
    plot_residual_components,
)
from src.visualize._epoch_visuals import save_epoch_visuals, create_epoch_grid
from src.visualize.cff_map import plot_cff_map


__all__ = [
    "nature_style",
    "save_figure",
    "add_colorbar",
    "annotate_significance",
    "compute_statistical_test",
    "format_pvalue",
    "create_nature_palette",
    "setup_figure",
    "SINGLE_COL",
    "DOUBLE_COL",
    "DPI",
    "MM_TO_INCH",
    "NATURE_PALETTE",
    "CMAP_PHASE",
    "CMAP_INTENSITY",
    "CMAP_ERROR_SIGNED",
    "CMAP_ERROR_ABS",
    "CMAP_CONTINUOUS",
    "CMAP_DIVERGING",
    "plot_training_curve",
    "plot_error_histogram",
    "plot_prediction_scatter",
    "plot_method_comparison",
    "plot_multiseed_comparison",
    "plot_convergence",
    "plot_phase_profile",
    "plot_phase_2d",
    "plot_loss_landscape",
    "plot_residual_analysis",
    "plot_residual_components",
    "save_epoch_visuals",
    "create_epoch_grid",
    "plot_cff_map",
]
