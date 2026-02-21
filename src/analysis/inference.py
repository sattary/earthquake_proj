import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from src.core.model import SpatialPINN
from src.core.physics import Physics
from src.data.loaders import GPSDataset
from src.data.velocity import VelocityModel
from src.data.transformers import CoordinateTransformer as Transformer


def load_inference_model(
    model_path: str | Path, device: torch.device, spatial_dim: int = 3
) -> SpatialPINN:
    """
    Loads a trained SpatialPINN model for inference.

    Rationale: Decouples model instantiation and state-dict loading from CLI
    entrypoints and visualization scripts.

    Args:
        model_path: Path to the .pth checkpoint.
        device: Target compute device.
        spatial_dim: Dimensionality of the model.

    Returns:
        SpatialPINN: The model in eval mode.
    """
    model = SpatialPINN(spatial_dim=spatial_dim, coupling_enabled=True).to(device)
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def compute_gps_azimuth_errors(
    model: SpatialPINN,
    device: torch.device,
    data_dir: str | Path = "data/kinematic_data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes principal strain azimuth predictions and errors across the GPS network.

    Rationale: Isolates batch data loading and forward-pass inference math.
    Ensures 'plot_*.py' visualizer functions act as pure sinks receiving
    numpy arrays, rather than complex data loaders.

    Args:
        model: Trained PINN.
        device: Computation device.
        data_dir: Directory containing 'gps_strain_*.csv'.

    Returns:
        Tuple of (theta_true, theta_pred, errors) in radians.
    """
    gps_files = list(Path(data_dir).glob("gps_strain_*.csv"))
    if not gps_files:
        raise ValueError(f"No GPS files found in {data_dir}")

    dataset = GPSDataset([str(f) for f in gps_files])

    with torch.no_grad():
        x = (
            torch.tensor(dataset.coords, dtype=torch.float32, device=device)
            .detach()
            .clone()
        )
        if model.spatial_dim == 3:
            z_surf = torch.full((x.shape[0], 1), -1.0, device=device)
            x = torch.cat([x, z_surf], dim=1)

        theta_true = dataset.azimuths_rad

        # SpatialPINN is expected to return (N, 9) in 3D:
        # [vx, vy, vz, sxx, syy, szz, sxy, syz, sxz]
        out_tensor = model(x)
        sxx = out_tensor[:, 3].cpu().numpy()
        syy = out_tensor[:, 4].cpu().numpy()
        sxy = out_tensor[:, 6].cpu().numpy()

        # Math angle inversion algorithm
        theta_pred = 0.5 * np.arctan2(2 * sxy, sxx - syy) + np.pi / 2

    diff = theta_pred - theta_true
    errors = np.arctan2(np.sin(diff), np.cos(diff))  # Wrapped error

    return theta_true, theta_pred, errors


def compute_cff_grid(
    model: SpatialPINN,
    depth_km: float,
    device: torch.device,
    velocity_file: str | Path = "data/Morteza_2023/Vel/Pwave.3D.txt",
    data_dir: str | Path = "data/kinematic_data",
    resolution: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the Coulomb Failure Function over a geographic grid at a specific depth.

    Rationale: Highly expensive matrix and meshgrid computations should reside in
    a designated computational analysis module.

    Args:
        model: Trained PINN.
        depth_km: Target depth slice.
        device: Compute device.
        velocity_file: Path to 3D velocity model (used for depth scaling bounds).
        data_dir: Directory containing GPS files (used to establish boundary box).
        resolution: X/Y grid resolution.

    Returns:
        Tuple of (Lon_mesh, Lat_mesh, CFF_values_mesh).
    """
    gps_files = list(Path(data_dir).glob("gps_strain_*.csv"))
    if not gps_files:
        raise ValueError(f"No GPS files found to establish boundaries in {data_dir}")

    dfs = [pd.read_csv(f) for f in gps_files]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    bounds = {
        "lon_min": df["longitude"].min(),
        "lon_max": df["longitude"].max(),
        "lat_min": df["latitude"].min(),
        "lat_max": df["latitude"].max(),
    }

    vel_model = VelocityModel(
        str(velocity_file),
        Transformer(df["latitude"].values, df["longitude"].values),
    )

    lon = np.linspace(bounds["lon_min"], bounds["lon_max"], resolution)
    lat = np.linspace(bounds["lat_min"], bounds["lat_max"], resolution)
    Lon, Lat = np.meshgrid(lon, lat)

    x_norm = (Lon.flatten() - bounds["lon_min"]) / (
        bounds["lon_max"] - bounds["lon_min"]
    )
    y_norm = (Lat.flatten() - bounds["lat_min"]) / (
        bounds["lat_max"] - bounds["lat_min"]
    )
    z_norm = (depth_km - vel_model.min_dep) / (
        vel_model.max_dep - vel_model.min_dep
    ) * 2.0 - 1.0

    pts = np.stack([x_norm, y_norm, np.full_like(x_norm, z_norm)], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device).requires_grad_(True)

    with torch.no_grad():
        cff_val, _ = Physics.coulomb_failure(
            model, pts_t, lambda_p=model.pore_pressure_ratio
        )
        cff_np = cff_val.cpu().numpy().reshape(resolution, resolution)

    return Lon, Lat, cff_np
