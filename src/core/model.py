import torch
import torch.nn as nn
import math
import numpy as np

from src.core.constants import A_PARAM_DEFAULT, LAMBDA_P_DEFAULT


class FourierFeature(nn.Module):
    """
    Fourier Feature Mapping to enable the MLP to learn high-frequency functions.
    Based on Tancik et al. (2020).
    """

    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x is (batch, input_dim) -> (batch, mapping_size)
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SpatialPINN(nn.Module):
    """
    Physics-Informed Neural Network for Lithospheric Stress Inversion.

    Inputs:
        x_coords: (batch, spatial_dim)

    Outputs:
        Predictions for Velocity and Stress fields.
        2D Case: [vx, vy, sxx, syy, sxy] (5 outputs)
        3D Case: [vx, vy, vz, sxx, syy, szz, sxy, syz, sxz] (9 outputs)
    """

    def __init__(
        self,
        spatial_dim=2,
        hidden_layers=[128, 128, 128, 128],
        fourier_scale=1.0,
        coupling_enabled=False,
    ):
        super().__init__()

        self.spatial_dim = spatial_dim

        # Determining output dimension
        if spatial_dim == 2:
            self.output_dim = 2 + 3  # vx, vy + sxx, syy, sxy
        elif spatial_dim == 3:
            self.output_dim = 3 + 6  # vx, vy, vz + sxx, syy, szz, sxy, syz, sxz
        else:
            raise ValueError(f"Spatial dim {spatial_dim} not supported.")

        # Embedding Layer
        self.fourier = FourierFeature(
            input_dim=spatial_dim, mapping_size=256, scale=fourier_scale
        )
        input_size = 256 * 2  # sin and cos

        # MLP Backbone
        layers = []
        for hidden in hidden_layers:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.Tanh())  # Tanh is standard for PINNs (smooth derivatives)
            input_size = hidden

        self.net = nn.Sequential(*layers)

        # Output Head
        self.head = nn.Linear(input_size, self.output_dim)

        # Learnable Coulomb-Dieterich coupling parameters
        self.coupling_enabled = coupling_enabled
        if coupling_enabled:
            # Log-space parameterization enforces positivity: A = exp(log_A)
            self.log_A = nn.Parameter(torch.tensor(math.log(A_PARAM_DEFAULT)))
            # Pore pressure ratio, clamped to [0, 1] in property
            self._lambda_p = nn.Parameter(torch.tensor(LAMBDA_P_DEFAULT))
            # Log-space background seismicity rate
            self.log_r0 = nn.Parameter(torch.tensor(0.0))

        # Initializing weights (Xavier usually good for Tanh)
        self._init_weights()

    @property
    def a_param(self):
        """Rate-and-state parameter A, always positive."""
        return torch.exp(self.log_A)

    @property
    def pore_pressure_ratio(self):
        """Pore pressure ratio lambda_p, clamped to [0, 1]."""
        return torch.clamp(self._lambda_p, 0.0, 1.0)

    @property
    def r0(self):
        """Background seismicity rate, always positive."""
        return torch.exp(self.log_r0)

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_normal_(self.head.weight)

    def forward(self, x):
        """
        Forward pass.
        x: (batch, spatial_dim)
        Returns:
            out: (batch, output_dim)
        """
        features = self.fourier(x)
        encoded = self.net(features)
        out = self.head(encoded)
        return out
