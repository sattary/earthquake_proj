"""
Synthetic Data Generator for 3D Physics-Informed Crustal Stress Inversion.

Generates:
1. Ground-truth stress and velocity fields on a grid.
2. Synthetic GPS strain-rate azimuths (with noise) at the surface.
3. Synthetic Earthquake Catalogs sampled from the Coulomb-Dieterich rate law.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from src.validation.synthetic_benchmark import AnalyticalBenchmark
from src.core.constants import RHO_CRUST, G_ACCEL, MU_FRICTION


class SyntheticDataGenerator:
    """Generates synthetic observatories from analytical tectonic models."""

    def __init__(
        self,
        domain_bounds: Optional[List[float]] = None,
        rho: float = RHO_CRUST,
        g: float = G_ACCEL,
        eta: float = 1e21,
    ):
        """
        Args:
            domain_bounds: [xmin, xmax, ymin, ymax, zmin, zmax] in meters.
                           Default is a 100km x 100km x 30km crustal block.
        """
        if domain_bounds is None:
            # 100x100 km surface, 30 km depth
            self.bounds = [0.0, 1e5, 0.0, 1e5, -30000.0, 0.0]
        else:
            self.bounds = domain_bounds

        self.rho = rho
        self.g = g
        self.benchmark = AnalyticalBenchmark(rho=rho, g=g, eta=eta)

    def generate_gps_data(
        self,
        regime: str,
        num_stations: int = 500,
        noise_std_deg: float = 2.0,
        out_path: str = "data/synthetic/gps_synthetic.csv",
    ) -> pd.DataFrame:
        """
        Generates noisy synthetic GPS strain-rate azimuths at the surface (z=0).

        Args:
            regime: Method name in AnalyticalBenchmark (e.g., 'simple_shear')
            num_stations: Number of random GPS stations to place.
            noise_std_deg: Standard deviation of Gaussian noise in degrees.
            out_path: File path to save the CSV.

        Returns:
            DataFrame of synthetic GPS data.
        """
        # Uniform sampling at surface z=0
        x = (
            torch.rand(num_stations) * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )
        y = (
            torch.rand(num_stations) * (self.bounds[3] - self.bounds[2])
            + self.bounds[2]
        )
        z = torch.zeros(num_stations)

        coords = torch.stack([x, y, z], dim=1)

        # Get analytical stress fields
        fields = getattr(self.benchmark, regime)(coords)
        s_xx = fields["s_xx"]
        s_yy = fields["s_yy"]
        s_xy = fields["s_xy"]

        # Maximum horizontal compressive strain-rate azimuth
        # theta = 0.5 * arctan2(2*s_xy, s_xx - s_yy)
        theta_rad = 0.5 * torch.atan2(2 * s_xy, s_xx - s_yy)
        theta_deg = torch.rad2deg(theta_rad)

        # Add Gaussian noise
        theta_noisy = theta_deg + torch.randn_like(theta_deg) * noise_std_deg

        # Wrap to [0, 180) degrees as azimuths are directionless lines
        theta_noisy = theta_noisy % 180.0

        df = pd.DataFrame(
            {
                "longitude": x.numpy(),
                "latitude": y.numpy(),
                "azimuth_value": theta_noisy.numpy(),
            }
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return df

    def generate_catalog(
        self,
        regime: str,
        num_events: int = 2000,
        lambda_p: float = 0.37,
        mu_f: float = MU_FRICTION,
        a_param: float = 0.01,
        r0: float = 1.0,
        out_path: str = "data/synthetic/catalog_synthetic.txt",
    ) -> pd.DataFrame:
        """
        Generates a synthetic earthquake catalog using Rejection Sampling from the
        Coulomb-Dieterich rate-and-state seismicity theory.

        Args:
            regime: Method name in AnalyticalBenchmark.
            num_events: Target number of earthquakes to sample.
            lambda_p: Pore fluid pressure ratio.
            mu_f: Coefficient of friction.
            a_param: Rate-and-state direct effect parameter.
            r0: Background reference seismicity rate.
            out_path: File path to save the catalog.

        Returns:
            DataFrame of earthquake spatial coordinates.
        """
        events_x, events_y, events_z = [], [], []

        # We need an estimate of the maximum possible rate for rejection sampling
        # Let's sample a large grid to find max rate
        grid_size = 50000
        x_grid = (
            torch.rand(grid_size) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        )
        y_grid = (
            torch.rand(grid_size) * (self.bounds[3] - self.bounds[2]) + self.bounds[2]
        )
        z_grid = (
            torch.rand(grid_size) * (self.bounds[5] - self.bounds[4]) + self.bounds[4]
        )
        coords_grid = torch.stack([x_grid, y_grid, z_grid], dim=1)

        rates_grid = self._compute_dieterich_rate(
            coords_grid, regime, lambda_p, mu_f, a_param, r0
        )
        max_rate = rates_grid.max().item() * 1.1  # 10% buffer

        # Rejection sampling loop
        batch_size = 10000
        while len(events_x) < num_events:
            x_cand = (
                torch.rand(batch_size) * (self.bounds[1] - self.bounds[0])
                + self.bounds[0]
            )
            y_cand = (
                torch.rand(batch_size) * (self.bounds[3] - self.bounds[2])
                + self.bounds[2]
            )
            z_cand = (
                torch.rand(batch_size) * (self.bounds[5] - self.bounds[4])
                + self.bounds[4]
            )
            coords_cand = torch.stack([x_cand, y_cand, z_cand], dim=1)

            rates_cand = self._compute_dieterich_rate(
                coords_cand, regime, lambda_p, mu_f, a_param, r0
            )

            # Accept if a random uniform [0, max_rate] is < rate(x)
            u = torch.rand(batch_size) * max_rate
            accepted = u < rates_cand

            x_acc = x_cand[accepted].tolist()
            y_acc = y_cand[accepted].tolist()
            z_acc = z_cand[accepted].tolist()

            events_x.extend(x_acc)
            events_y.extend(y_acc)
            events_z.extend(z_acc)

        # Truncate to exact number requested
        df = pd.DataFrame(
            {
                "long": events_x[:num_events],
                "lat": events_y[:num_events],
                "fd": events_z[:num_events],
                "mw_unified": np.random.uniform(
                    4.0, 7.0, num_events
                ),  # Arbitrary magnitudes
            }
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # Format similar to historical_Eq.txt, but using standard commas
        # as loaders.py uses default pd.read_csv()
        df.to_csv(out_path, index=False)
        return df

    def _compute_dieterich_rate(
        self,
        coords: torch.Tensor,
        regime: str,
        lambda_p: float,
        mu_f: float,
        a_param: float,
        r0: float,
    ) -> torch.Tensor:
        """Internal helper to compute the Coulomb-Dieterich seismicity rate."""
        fields = getattr(self.benchmark, regime)(coords)
        s_xx = fields["s_xx"]
        s_yy = fields["s_yy"]
        s_zz = fields["s_zz"]
        s_xy = fields["s_xy"]
        s_yz = fields["s_yz"]
        s_xz = fields["s_xz"]

        # Build symmetric 3x3 stress tensor
        stress_tensor = torch.stack(
            [
                torch.stack([s_xx, s_xy, s_xz], dim=-1),
                torch.stack([s_xy, s_yy, s_yz], dim=-1),
                torch.stack([s_xz, s_yz, s_zz], dim=-1),
            ],
            dim=-2,
        )

        # Eigenvalues to find principal stresses
        eigvals = torch.linalg.eigvalsh(stress_tensor)
        s3 = eigvals[:, 0]  # Most compressive
        s1 = eigvals[:, 2]  # Least compressive

        tau_max = (s1 - s3) / 2.0
        sigma_n = (s1 + s3) / 2.0

        # Pore fluid pressure: Pf = lambda_p * rho * g * depth
        p_f = lambda_p * self.rho * self.g * torch.abs(coords[:, 2])

        # Coulomb Failure Function
        cff = tau_max - mu_f * (sigma_n - p_f)

        # Dieterich Rate
        a_sigma = a_param * torch.abs(sigma_n).clamp(min=1e3)
        rate = r0 * torch.exp((cff / a_sigma).clamp(max=20.0, min=-20.0))

        return rate

    def generate_ground_truth_grid(
        self,
        regime: str,
        resolution: int = 20,
        out_path: str = "data/synthetic/truth.pt",
    ) -> dict[str, torch.Tensor]:
        """
        Generates a dense 3D grid of ground truth stresses for error quantification.
        """
        x_lin = torch.linspace(self.bounds[0], self.bounds[1], resolution)
        y_lin = torch.linspace(self.bounds[2], self.bounds[3], resolution)
        z_lin = torch.linspace(self.bounds[4], self.bounds[5], resolution)

        grid_x, grid_y, grid_z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
        coords = torch.stack(
            [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1
        )

        fields = getattr(self.benchmark, regime)(coords)
        fields["coords"] = coords

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(fields, out_path)
        return fields
