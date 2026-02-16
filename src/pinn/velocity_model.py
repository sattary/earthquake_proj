import numpy as np
import scipy.interpolate
import torch
import pandas as pd
from src.pinn.utils import CoordinateTransformer


class VelocityModel:
    """
    Handles 3D Velocity Structure (Vp) and derived material properties.
    Loads 'Pwave.3D.txt' and provides interpolation for (x, y, z).
    """

    def __init__(self, velocity_file, transformer: CoordinateTransformer):
        """
        Args:
            velocity_file (str): Path to Pwave.3D.txt
            transformer (CoordinateTransformer): To convert model Lat/Lon to UTM/Normalized.
        """
        print(f"Loading Velocity Model from {velocity_file}...")
        # Load and parse
        # Columns: long(degree) lat(degree) dep(km) Vp(km/s)
        df = pd.read_csv(velocity_file, sep=r"\s+")

        lons = df.iloc[:, 0].values
        lats = df.iloc[:, 1].values
        deps = df.iloc[:, 2].values  # Depth in km (positive down?) check file
        vps = df.iloc[:, 3].values

        # Convert Model Coords to Normalized Coords
        # 1. Lat/Lon -> UTM -> Normalized (x, y)
        norm_xy = transformer.to_normalized(lats, lons)  # Returns Tensor
        x_norm = norm_xy[:, 0].numpy()
        y_norm = norm_xy[:, 1].numpy()

        # 2. Depth -> Normalized z
        # We need to define Z-bounds for the PINN.
        # Let's assume the PINN domain matches the velocity model depth range.
        self.min_dep = deps.min()
        self.max_dep = deps.max()

        # Simple Linear Normalization for Z to [-1, 1] (or [0, 1])
        # Let's use [-1, 1] to match X/Y
        # z_norm = 2 * (dep - min) / (max - min) - 1
        z_norm = 2 * (deps - self.min_dep) / (self.max_dep - self.min_dep) - 1

        # Prepare Interpolator
        # (x_norm, y_norm, z_norm) -> Vp
        points = np.stack([x_norm, y_norm, z_norm], axis=1)

        print("Building 3D Interpolator (Nearest for speed/robustness)...")
        # NearestNDInterpolator is robust for scattered data.
        # LinearNDInterpolator is better but can NaN outside convex hull.
        self.vp_interp = scipy.interpolate.NearestNDInterpolator(points, vps)

        print(f"Velocity Model Ready. Depth Range: {self.min_dep}-{self.max_dep} km")

    def get_material_properties(self, x_norm, y_norm, z_norm):
        """
        Query Vp and return Rho and Mu.
        Args:
            x_norm, y_norm, z_norm: Tensors (N,) or (N,1)
        Returns:
            rho (Tensor): Density in kg/m^3
            mu (Tensor): Shear Modulus in Pa
        """
        if isinstance(x_norm, torch.Tensor):
            xn = x_norm.detach().cpu().numpy().flatten()
            yn = y_norm.detach().cpu().numpy().flatten()
            zn = z_norm.detach().cpu().numpy().flatten()
        else:
            xn = x_norm.flatten()
            yn = y_norm.flatten()
            zn = z_norm.flatten()

        # Query Vp
        vp = self.vp_interp(np.stack([xn, yn, zn], axis=1))  # km/s

        # Empirical Relations
        # Brocher (2005) or Birch's Law for density
        # rho = 1.6612 * Vp - 0.4721 * Vp**2 + 0.0671 * Vp**3 - 0.0043 * Vp**4 + 0.000106 * Vp**5
        # Simplified Birch: rho = 0.32 * Vp + 0.77 (Vp in km/s, rho in g/cm3)
        # Let's use simple scaling for stability: rho ~ 2700

        # Vp is km/s -> m/s = Vp * 1000
        vp_m = vp * 1000.0

        # Naive Density: 2700 kg/m^3 baseline, scaled slightly by Vp
        # rho = 2700 + (vp - 6.0)*300
        rho = 2500.0 + (vp - 5.0) * 200.0

        # Shear Modulus mu = rho * Vs^2
        # Assume Poisson solid (lambda = mu) -> Vp = sqrt(3)*Vs -> Vs = Vp / sqrt(3)
        vs_m = vp_m / 1.732

        mu = rho * (vs_m**2)

        return {
            "vp": torch.tensor(vp, dtype=torch.float32),
            "rho": torch.tensor(rho, dtype=torch.float32),
            "mu": torch.tensor(mu, dtype=torch.float32),
        }

    def normalize_z(self, depth_km):
        # Helper to convert physical depth to normalized z
        return 2 * (depth_km - self.min_dep) / (self.max_dep - self.min_dep) - 1

    def denormalize_z(self, z_norm_val):
        # Helper to convert normalized z to physical depth
        return (z_norm_val + 1) * (self.max_dep - self.min_dep) / 2 + self.min_dep
