import torch
import numpy as np
import pytest
from src.pinn.utils import CoordinateTransformer


def test_coordinate_transformer_reversibility():
    # Sample Lat/Lon in Iran (Alborz region)
    lats = np.array([35.5, 36.0, 36.5])
    lons = np.array([50.5, 51.0, 51.5])

    transformer = CoordinateTransformer(lats, lons)

    # Lat/Lon -> Normalized
    coords_norm = transformer.to_normalized(lats, lons)
    assert coords_norm.shape == (3, 2)
    assert torch.all(coords_norm >= -1.1) and torch.all(coords_norm <= 1.1)

    # Normalized -> Lat/Lon
    lats_rev, lons_rev = transformer.to_physical(coords_norm)

    # Check if they are close
    assert np.allclose(lats, lats_rev, atol=1e-5)
    assert np.allclose(lons, lons_rev, atol=1e-5)


def test_azimuth_math_conversion():
    # GPS Azimuth (CW from North)
    # 0 -> 90 math (North)
    # 90 -> 0 math (East)
    # 180 -> -90 math (South)
    gps_az = torch.tensor([0.0, 90.0, 180.0]) * (torch.pi / 180.0)
    math_az = torch.pi / 2 - gps_az

    assert torch.allclose(math_az[0], torch.tensor(torch.pi / 2))
    assert torch.allclose(math_az[1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(math_az[2], torch.tensor(-torch.pi / 2))
