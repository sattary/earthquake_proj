"""
Multi-Epoch CFF Progression Visualizer.
Generates a horizontal grid showing the Coulomb Failure Function (CFF) map
at multiple checkpoints to demonstrate spatiotemporal learning progression.
"""

from pathlib import Path
from typing import List
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.core.model import SpatialPINN
from src.core.physics import Physics
from src.visualize.style import nature_style, save_figure


def plot_cff_progression(
    checkpoint_paths: List[str], depth: float, out_path: str, device: str = "cpu"
):
    """
    Plot the CFF map evolution across multiple checkpoints.

    Args:
        checkpoint_paths: List of absolute or relative paths to .pth files.
        depth: The depth slice to query CFF (in km).
        out_path: Where to save the resulting multipanel figure.
        device: CPU or CUDA. Defaults to 'cpu' for decoupling.
    """
    # Attempt to load Transformer bounds from data
    gps_files = list(Path("data/kinematic_data").glob("gps_strain_*.csv"))
    if not gps_files:
        print(
            "Warning: No GPS files found. Synthesizing dummy bounds for CFF progression."
        )
        bounds = {"lon_min": 44.0, "lon_max": 64.0, "lat_min": 25.0, "lat_max": 40.0}
    else:
        dfs = [pd.read_csv(f) for f in gps_files]
        df = pd.concat(dfs, ignore_index=True)
        bounds = {
            "lon_min": df["lon"].min(),
            "lon_max": df["lon"].max(),
            "lat_min": df["lat"].min(),
            "lat_max": df["lat"].max(),
        }

    res = 100
    lon = np.linspace(bounds["lon_min"], bounds["lon_max"], res)
    lat = np.linspace(bounds["lat_min"], bounds["lat_max"], res)
    Lon, Lat = np.meshgrid(lon, lat)

    x_norm = (Lon.flatten() - bounds["lon_min"]) / (
        bounds["lon_max"] - bounds["lon_min"]
    )
    y_norm = (Lat.flatten() - bounds["lat_min"]) / (
        bounds["lat_max"] - bounds["lat_min"]
    )

    # Normally velocity model defines depth bounds, using typical limits 0 to 30 km
    z_norm = (depth - 0.0) / (30.0 - 0.0) * 2.0 - 1.0

    pts = np.stack([x_norm, y_norm, np.full_like(x_norm, z_norm)], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device).requires_grad_(True)

    n_panels = len(checkpoint_paths)
    if n_panels == 0:
        print("No checkpoints provided for CFF progression.")
        return

    # Use nature_style dimensions but scale width by panels
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))
    if n_panels == 1:
        axes = [axes]

    vmin, vmax = -5, 5  # Standardized limits for reliable visual comparison

    with nature_style():
        for i, ckpt_path in enumerate(checkpoint_paths):
            ax = axes[i]
            if not Path(ckpt_path).exists():
                ax.text(0.5, 0.5, "Checkpoint Missing", ha="center", va="center")
                ax.axis("off")
                continue

            model = SpatialPINN(spatial_dim=3, coupling_enabled=True).to(device)
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            except Exception as e:
                print(f"Failed to load {ckpt_path}: {e}")
                ax.text(0.5, 0.5, "Load Error", ha="center", va="center")
                ax.axis("off")
                continue

            model.eval()

            with torch.no_grad():
                cff_val, _ = Physics.coulomb_failure(
                    model, pts_t, lambda_p=model.pore_pressure_ratio
                )
                cff_np = cff_val.cpu().numpy().reshape(res, res)

            # Get epoch from filename, e.g., checkpoint_epoch_1000.pth
            name_parts = Path(ckpt_path).stem.split("_")
            epoch_str = name_parts[-1] if name_parts[-1].isdigit() else "Final"

            im = ax.imshow(
                cff_np,
                extent=[
                    bounds["lon_min"],
                    bounds["lon_max"],
                    bounds["lat_min"],
                    bounds["lat_max"],
                ],
                origin="lower",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"Epoch {epoch_str}", fontsize=10)
            ax.set_xlabel("Longitude")
            if i == 0:
                ax.set_ylabel("Latitude")
            else:
                ax.set_yticks([])

        plt.tight_layout()
        # Add colorbar globally
        cbar = fig.colorbar(
            im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.15
        )
        cbar.set_label("CFF (MPa)")

        save_figure(fig, out_path)
        print(f"Generated CFF progression at {out_path}")
