"""
3D Volumetric Hero Visualization.
Creates a publication-quality interactive or static 3D block model
showing velocity structure and volumetric residuals via PyVista.
"""

from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pyvista as pv

from src.core.model import SpatialPINN
from src.data.transformers import CoordinateTransformer


def plot_3d_fault_residuals(model_path: str, out_path: str, device: str = "cpu"):
    """
    Generates a 3D block representation of the PINN inversion.
    Saves as an interactive HTML and a static high-res screenshot.
    """
    gps_files = list(Path("data/kinematic_data").glob("gps_strain_*.csv"))
    if not gps_files:
        print("Error: No GPS files found for 3D Hero Plot.")
        return

    dfs = [pd.read_csv(f) for f in gps_files]
    df = pd.concat(dfs, ignore_index=True)

    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    depth_min, depth_max = 0.0, 30.0

    transformer = CoordinateTransformer(df["latitude"].values, df["longitude"].values)

    # 1. Create 3D Structured Grid for the crustal block
    res_x, res_y, res_z = 30, 30, 15
    lons = np.linspace(lon_min, lon_max, res_x)
    lats = np.linspace(lat_min, lat_max, res_y)
    deps = np.linspace(depth_min, depth_max, res_z)

    Lons, Lats, Deps = np.meshgrid(lons, lats, deps, indexing="ij")

    x_norm = (Lons.flatten() - lon_min) / (lon_max - lon_min)
    y_norm = (Lats.flatten() - lat_min) / (lat_max - lat_min)
    z_norm = (Deps.flatten() - depth_min) / (depth_max - depth_min) * 2.0 - 1.0

    pts = np.stack([x_norm, y_norm, z_norm], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)

    model = SpatialPINN(spatial_dim=3, coupling_enabled=True).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return

    with torch.no_grad():
        out_tensor = model(pts_t)
        # We plot shear stress as the volumetric scalar
        sxy = out_tensor[:, 6].cpu().numpy()
        syz = out_tensor[:, 7].cpu().numpy()
        sxz = out_tensor[:, 8].cpu().numpy()

        # Simple proxy for shear intensity
        shear_intensity = np.sqrt(sxy**2 + syz**2 + sxz**2)

    # Building PyVista Grid
    grid = pv.StructuredGrid(Lons, Lats, Deps)
    grid["Shear Intensity (MPa)"] = shear_intensity

    # 2. Extract surface GPS errors for "heat cloud" or spheres
    lons_obs = df["longitude"].values
    lats_obs = df["latitude"].values
    theta_true_rad = np.deg2rad(df["azimuth_value"].values)

    norm_obs = transformer.to_normalized(lats_obs, lons_obs)
    pts_obs = np.hstack([norm_obs, np.full((len(lons_obs), 1), -1.0)])
    pts_obs_t = torch.tensor(pts_obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        out_obs = model(pts_obs_t)
        sxx_obs = out_obs[:, 3].cpu().numpy()
        syy_obs = out_obs[:, 4].cpu().numpy()
        sxy_obs = out_obs[:, 6].cpu().numpy()
        theta_pred_rad = 0.5 * np.arctan2(2 * sxy_obs, sxx_obs - syy_obs) + np.pi / 2

    diff = theta_pred_rad - theta_true_rad
    error_deg = np.rad2deg(np.abs(np.arctan2(np.sin(diff), np.cos(diff))))

    # Create PyVista PointCloud for GPS observations
    gps_z = np.zeros_like(lons_obs)  # Plot at depth=0
    gps_points = pv.PolyData(np.column_stack((lons_obs, lats_obs, gps_z)))
    gps_points["Azimuth Error (deg)"] = error_deg

    # 3. Visualization Rendering
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")

    # Add volumetric rendering of the crust (semi-transparent)
    volume_outline = grid.outline()
    plotter.add_mesh(volume_outline, color="black", line_width=2)

    # Slices for internal view
    slices = grid.slice_orthogonal()
    plotter.add_mesh(
        slices, scalars="Shear Intensity (MPa)", cmap="viridis", opacity=0.8
    )

    # Add GPS spheres sized/colored by error
    # Exaggerate Z to make points visible above the mesh crust
    glyphs = gps_points.glyph(
        scale="Azimuth Error (deg)", geom=pv.Sphere(), factor=0.01
    )
    plotter.add_mesh(
        glyphs, scalars="Azimuth Error (deg)", cmap="hot", render_points_as_spheres=True
    )

    plotter.add_axes()

    # Save as static high-res image
    out_img = out_path.replace(".html", ".png")
    plotter.screenshot(out_img, transparent_background=False, window_size=[1920, 1080])
    print(f"Generated 3D Hero Static PNG at {out_img}")

    # Save as interactive HTML for Supplementary Material
    out_html = out_path.replace(".png", ".html")
    try:
        plotter.export_html(out_html)
        print(f"Generated 3D Hero Interactive HTML at {out_html}")
    except Exception as e:
        print(f"Could not export HTML (requires ipyvtklink or panel): {e}")


if __name__ == "__main__":
    import os

    os.makedirs("results/figs", exist_ok=True)
    plot_3d_fault_residuals(
        model_path="checkpoints/final_model.pth",
        out_path="results/figs/volumetric_hero.html",
    )
