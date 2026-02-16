import torch
import torch.nn as nn
import numpy as np


class FourierFeature(nn.Module):
    """
    Fourier Feature Mapping to enable the MLP to learn high-frequency functions.
    Based on Tancik et al. (2020).
    """

    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.B = nn.Parameter(
            torch.randn(input_dim, mapping_size) * scale, requires_grad=False
        )

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
        self, spatial_dim=2, hidden_layers=[128, 128, 128, 128], fourier_scale=1.0
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

        # Initializing weights (Xavier usually good for Tanh)
        self._init_weights()

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
