"""
Smoke tests for the visualization suite.
"""

import numpy as np
import pytest


class TestStyleImports:
    """Test that all style components can be imported."""

    def test_import_style(self):
        from src.visualize import (
            nature_style,
            save_figure,
            SINGLE_COL,
            DOUBLE_COL,
            NATURE_PALETTE,
        )

        assert SINGLE_COL > 0
        assert DOUBLE_COL > SINGLE_COL
        assert "blue" in NATURE_PALETTE

    def test_nature_style_context(self):
        from src.visualize import nature_style

        with nature_style():
            import matplotlib as mpl

            assert mpl.rcParams.get("figure.dpi") == 300

    def test_palette_creation(self):
        from src.visualize import create_nature_palette

        colors = create_nature_palette(6)
        assert len(colors) == 6
        assert all(isinstance(c, str) for c in colors)

    def test_format_pvalue(self):
        from src.visualize import format_pvalue

        assert format_pvalue(0.0001) == "***"
        assert format_pvalue(0.005) == "**"
        assert format_pvalue(0.03) == "*"
        assert format_pvalue(0.2) == "ns"


class TestPlotFunctions:
    """Test that plot functions can be imported."""

    def test_import_training_curve(self):
        from src.visualize import plot_training_curve

        assert callable(plot_training_curve)

    def test_import_error_histogram(self):
        from src.visualize import plot_error_histogram

        assert callable(plot_error_histogram)

    def test_import_prediction_scatter(self):
        from src.visualize import plot_prediction_scatter

        assert callable(plot_prediction_scatter)

    def test_import_method_comparison(self):
        from src.visualize import plot_method_comparison

        assert callable(plot_method_comparison)

    def test_import_convergence(self):
        from src.visualize import plot_convergence

        assert callable(plot_convergence)

    def test_import_phase_profile(self):
        from src.visualize import plot_phase_profile

        assert callable(plot_phase_profile)


class TestStatisticalFunctions:
    """Test statistical utilities."""

    def test_compute_statistical_test(self):
        from src.visualize import compute_statistical_test

        group1 = np.random.randn(100)
        group2 = np.random.randn(100) + 0.5

        stat, pval = compute_statistical_test(group1, group2, "ttest")
        assert isinstance(stat, float)
        assert isinstance(pval, float)

        stat, pval = compute_statistical_test(group1, group2, "mannwhitney")
        assert isinstance(pval, float)


class TestPlotExecution:
    """Test that plots can execute without error."""

    def test_error_histogram_execution(self, tmp_path):
        from src.visualize import plot_error_histogram

        errors = np.random.randn(1000)

        out_file = tmp_path / "error_hist"
        plot_error_histogram(errors, out_path=str(out_file))

        assert (tmp_path / "error_hist.png").exists()

    def test_prediction_scatter_execution(self, tmp_path):
        from src.visualize import plot_prediction_scatter

        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1

        out_file = tmp_path / "scatter"
        plot_prediction_scatter(y_true, y_pred, out_path=str(out_file))

        assert (tmp_path / "scatter.png").exists()

    def test_phase_profile_execution(self, tmp_path):
        from src.visualize import plot_phase_profile

        x = np.linspace(0, 100, 50)
        phase = np.sin(x / 10) * np.pi

        out_file = tmp_path / "phase"
        plot_phase_profile(x, phase, out_path=str(out_file))

        assert (tmp_path / "phase.png").exists()
