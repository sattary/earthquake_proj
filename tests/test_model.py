import torch
import pytest
from src.pinn.model import SpatialPINN


def test_spatial_pinn_2d_shape():
    model = SpatialPINN(spatial_dim=2)
    x = torch.randn(10, 2)
    out = model(x)
    assert out.shape == (10, 5)  # vx, vy, sxx, syy, sxy


def test_spatial_pinn_3d_shape():
    model = SpatialPINN(spatial_dim=3)
    x = torch.randn(10, 3)
    out = model(x)
    assert out.shape == (10, 9)  # vx, vy, vz, sxx, syy, szz, sxy, syz, sxz


def test_fourier_mapping():
    model = SpatialPINN(spatial_dim=2, fourier_scale=10.0)
    x1 = torch.tensor([[0.0, 0.0]])
    x2 = torch.tensor([[0.1, 0.1]])

    # Check that fourier features are different for different inputs
    feat1 = model.fourier(x1)
    feat2 = model.fourier(x2)
    assert not torch.allclose(feat1, feat2)
