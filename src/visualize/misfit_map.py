"""
Geographic Misfit Map Visualizer.
Plots observed GPS vectors (black) against PINN predicted vectors (red)
overlaid on a continuous geographic heatmap of the scalar misfit error.
"""

from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.core.model import SpatialPINN
from src.visualize.style import nature_style, save_figure
from src.data.transformers import CoordinateTransformer


def plot_misfit_map(model_path: str, out_path: str, device: str = "cpu"):
    """
    Generate a geographic map of Observed vs Predicted GPS vectors.
    """
    gps_files = list(Path("data/kinematic_data").glob("gps_strain_*.csv"))
    if not gps_files:
        print("Error: No GPS files found for Misfit Map.")
        return

    # Load and aggregate data
    dfs = [pd.read_csv(f) for f in gps_files]
    df = pd.concat(dfs, ignore_index=True)

    # Needs columns: longitude, latitude, azimuth_value
    # Assuming standard velocity logic or just azimuths for now.
    # The PINN predicts sxx, syy, sxy. We derive principal stress directions.

    lons = df["longitude"].values
    lats = df["latitude"].values
    theta_true_deg = df["azimuth_value"].values
    theta_true_rad = np.deg2rad(theta_true_deg)

    transformer = CoordinateTransformer(lats, lons)
    normalized_coords = transformer.to_normalized(lats, lons)

    model = SpatialPINN(spatial_dim=3, coupling_enabled=True).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return

    with torch.no_grad():
        # Evaluate model at GPS surface locations (z=0 -> normalized z=-1 assuming depth [0, 30])
        z_norm = np.full((len(lons), 1), -1.0)
        pts = np.hstack([normalized_coords, z_norm])
        pts_t = torch.tensor(pts, dtype=torch.float32, device=device)

        out_tensor = model(pts_t)
        sxx = out_tensor[:, 3].cpu().numpy()
        syy = out_tensor[:, 4].cpu().numpy()
        sxy = out_tensor[:, 6].cpu().numpy()

        # Principal stress direction (SHmax azimuth)
        theta_pred_rad = 0.5 * np.arctan2(2 * sxy, sxx - syy) + np.pi / 2

    # Calculate Angular Misfit
    diff = theta_pred_rad - theta_true_rad
    error_rad = np.arctan2(np.sin(diff), np.cos(diff))
    error_deg = np.rad2deg(np.abs(error_rad))

    # Vector fields for plotting (Length arbitrary, direction matters)
    u_true = np.sin(theta_true_rad)
    v_true = np.cos(theta_true_rad)

    u_pred = np.sin(theta_pred_rad)
    v_pred = np.cos(theta_pred_rad)

    fig, ax = plt.subplots(figsize=(8, 6))

    with nature_style():
        # Scatter for heatmap background
        sc = ax.scatter(
            lons, lats, c=error_deg, cmap="Reds", s=15, alpha=0.8, edgecolors="none"
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Azimuth Misfit (Degrees)")

        # Subsample for quiver plot to avoid overcrowding
        step = max(1, len(lons) // 100)

        ax.quiver(
            lons[::step],
            lats[::step],
            u_true[::step],
            v_true[::step],
            color="black",
            scale=20,
            label="Observed (True)",
            alpha=0.7,
        )
        ax.quiver(
            lons[::step],
            lats[::step],
            u_pred[::step],
            v_pred[::step],
            color="red",
            scale=20,
            label="PINN Predicted",
            alpha=0.9,
        )

        ax.set_title("Maximum Horizontal Stress Directions: Observed vs Predicted")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_figure(fig, out_path)
        print(f"Generated Misfit Map at {out_path}")


if __name__ == "__main__":
    plot_misfit_map(
        model_path="checkpoints/final_model.pth", out_path="results/figs/misfit_map.png"
    )
