"""
Analytical Benchmarks for 3D Physics-Informed Crustal Stress Inversion.

Provides exact closed-form solutions for Anderson-Thompson tectonic faulting states.
These ground-truth models are critical for the Synthetic Recovery Validation phase,
resolving the magnitude ambiguity by proving the PINN's theoretical well-posedness.
"""

import torch
from typing import Dict


class AnalyticalBenchmark:
    """
    Ground-truth analytical models for 3D tectonic stress and velocity fields.
    Satisfies both the momentum balance (PDE) and viscous constitutive laws.
    """

    def __init__(self, rho: float = 2700.0, g: float = 9.81, eta: float = 1e21):
        """
        Initialize the physical constants for the synthetic crust.

        Args:
            rho: Average crustal density (kg/m^3)
            g: Gravitational acceleration (m/s^2)
            eta: Effective viscosity (Pa·s)
        """
        self.rho = rho
        self.g = g
        self.eta = eta

    def lithostatic(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes a purely lithostatic (hydrostatic) stress state driven by gravity.

        Args:
            coords: (N, 3) tensor of [x, y, z] spatial coordinates.
                    z is depth, assumed negative upwards.
        Returns:
            Dictionary of tensor fields for velocity and stress.
        """
        z = coords[:, 2]
        zeros = torch.zeros_like(z)

        # Lithostatic pressure: P = rho * g * z (where z is positive depth)
        sigma_zz = -self.rho * self.g * z
        sigma_xx = sigma_zz
        sigma_yy = sigma_zz

        # Velocity is identically zero for pure lithostatic state
        vx = vy = vz = zeros

        return {
            "v_x": vx,
            "v_y": vy,
            "v_z": vz,
            "s_xx": sigma_xx,
            "s_yy": sigma_yy,
            "s_zz": sigma_zz,
            "s_xy": zeros,
            "s_yz": zeros,
            "s_xz": zeros,
        }

    def simple_shear(
        self, coords: torch.Tensor, tau_max: float = 1e7
    ) -> Dict[str, torch.Tensor]:
        """
        Strike-slip simple shear regime (e.g., San Andreas or North Anatolian style).
        Imposes a constant horizontal shear stress overlaid on lithostatic pressure.

        Args:
            coords: (N, 3) tensor of spatial coordinates.
            tau_max: Maximum shear stress magnitude (Pa). Default 10 MPa.
        """
        y = coords[:, 1]
        z = coords[:, 2]
        zeros = torch.zeros_like(y)

        sigma_zz = -self.rho * self.g * z
        sigma_xx = sigma_zz
        sigma_yy = sigma_zz

        s_xy = torch.full_like(y, tau_max)
        s_yz = zeros
        s_xz = zeros

        # Viscous relationship: tau_xy = eta * (dv_x/dy + dv_y/dx)
        # Distributing shear entirely into v_x gives: dv_x/dy = tau_max / eta
        vx = (tau_max / self.eta) * y
        vy = zeros
        vz = zeros

        return {
            "v_x": vx,
            "v_y": vy,
            "v_z": vz,
            "s_xx": sigma_xx,
            "s_yy": sigma_yy,
            "s_zz": sigma_zz,
            "s_xy": s_xy,
            "s_yz": s_yz,
            "s_xz": s_xz,
        }

    def uniaxial_compression(
        self, coords: torch.Tensor, sigma_max: float = 1e8
    ) -> Dict[str, torch.Tensor]:
        """
        Reverse/Thrust faulting regime (e.g., Zagros collision zone).
        Imposes uniaxial tectonic compression along the X-axis overlaid on lithostatic pressure.

        Args:
            coords: (N, 3) tensor of spatial coordinates.
            sigma_max: Differential tectonic compressive stress (Pa). Default 100 MPa.
        """
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        zeros = torch.zeros_like(x)

        sigma_zz = -self.rho * self.g * z
        sigma_yy = sigma_zz
        # Compression is negative in standard geomechanics sign convention
        sigma_xx = sigma_zz - sigma_max

        # Deviatoric stresses: (assuming incompressible, div(v) = 0)
        # P = (sigma_xx + sigma_yy + sigma_zz) / 3 = sigma_zz - sigma_max/3
        # tau_xx = sigma_xx - P = -2/3 sigma_max
        # tau_yy = sigma_yy - P = 1/3 sigma_max
        # tau_zz = sigma_zz - P = 1/3 sigma_max

        # Viscous equations: tau_ii = 2 * eta * dv_i/dx_i
        vx = (-sigma_max / (3.0 * self.eta)) * x
        vy = (sigma_max / (6.0 * self.eta)) * y
        vz = (sigma_max / (6.0 * self.eta)) * z

        return {
            "v_x": vx,
            "v_y": vy,
            "v_z": vz,
            "s_xx": sigma_xx,
            "s_yy": sigma_yy,
            "s_zz": sigma_zz,
            "s_xy": zeros,
            "s_yz": zeros,
            "s_xz": zeros,
        }
