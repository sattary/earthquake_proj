"""
Per-epoch training visualization utilities.

Quick visual debugging during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.visualize.style import nature_style, save_figure, SINGLE_COL


def save_epoch_visuals(
    model,
    batch,
    epoch: int,
    out_dir: str,
    max_samples: int = 4,
) -> None:
    """
    Save per-epoch visualization of model predictions.

    Args:
        model: The PINN model
        batch: Input batch (coordinates, targets)
        epoch: Current epoch number
        out_dir: Output directory for visualizations
        max_samples: Maximum samples to visualize
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        coords = batch[0]
        targets = batch[1] if len(batch) > 1 else None

        with torch.no_grad():
            predictions = model(coords)

        coords = coords.cpu().numpy()[:max_samples]
        predictions = predictions.cpu().numpy()[:max_samples]

        if targets is not None:
            targets = targets.cpu().numpy()[:max_samples]

        fig, axes = plt.subplots(1, 3, figsize=(SINGLE_COL * 2, SINGLE_COL * 0.4))

        with nature_style():
            if coords.ndim > 1:
                x = coords[:, 0]
            else:
                x = coords

            axes[0].plot(
                x,
                predictions[:, 0] if predictions.ndim > 1 else predictions,
                "b-",
                lw=1,
            )
            axes[0].set_title("Prediction", fontsize=9)

            if targets is not None:
                axes[1].plot(
                    x, targets[:, 0] if targets.ndim > 1 else targets, "r-", lw=1
                )
                axes[1].set_title("Target", fontsize=9)

                if targets.ndim > 1 and predictions.ndim > 1:
                    error = predictions[:, 0] - targets[:, 0]
                else:
                    error = predictions - targets
                axes[2].plot(x, error, "k-", lw=1)
                axes[2].set_title("Error", fontsize=9)
            else:
                axes[1].plot(
                    x,
                    predictions[:, 0] if predictions.ndim > 1 else predictions,
                    "r-",
                    lw=1,
                )
                axes[1].set_title("Prediction", fontsize=9)

            for ax in axes:
                ax.grid(True, alpha=0.3)

            plt.suptitle(f"Epoch {epoch}")
            plt.tight_layout()

            save_figure(fig, str(out_path / f"epoch_{epoch:05d}"))

    except Exception as e:
        print(f"Warning: Could not save epoch visuals: {e}")


def create_epoch_grid(
    epochs: list[int],
    metrics: dict,
    out_path: Optional[str] = None,
) -> None:
    """
    Create a grid showing metrics across selected epochs.

    Args:
        epochs: List of epoch numbers to show
        metrics: Dict of {epoch: {metric_name: value}}
        out_path: Output path for figure (optional)
    """
    n_metrics = len(list(metrics.values())[0]) if metrics else 0

    fig, axes = plt.subplots(
        1, n_metrics, figsize=(SINGLE_COL * n_metrics, SINGLE_COL * 0.4)
    )

    if n_metrics == 1:
        axes = [axes]

    with nature_style():
        metric_names = list(list(metrics.values())[0].keys())

        for i, metric in enumerate(metric_names):
            values = [metrics.get(e, {}).get(metric, np.nan) for e in epochs]
            axes[i].plot(epochs, values, "o-", lw=1, markersize=4, color="#0072B2")
            axes[i].set_title(metric, fontsize=9)
            axes[i].set_xlabel("Epoch", fontsize=9)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()
        plt.close()
