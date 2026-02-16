import torch
import pytest
from src.pinn.physics import Physics
from src.pinn.model import SpatialPINN


def test_momentum_balance_constant_field():
    # In a constant velocity/stress field, residuals should be zero (excluding gravity)
    # Actually res_z = div_z + rho*g.
    # If stress is constant, div_z = 0, so res_z = rho*g.
    # But residuals are normalized by 1/(rho*g).
    # So for constant stress, res_z should be 1.0.

    model = SpatialPINN(spatial_dim=3)
    # Set all weights to zero and biases to constant so derivatives are zero
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0)

    x = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

    rx, ry, rz = Physics.momentum_balance_3d(model, x, rho=2700.0, g=9.81, S0=1e8)

    # Static case (all derivatives 0):
    # res_z = (div_z + rho*g) / (rho*g) = (0 + 26487) / 26487 = 1.0
    assert torch.allclose(rx, torch.tensor(0.0))
    assert torch.allclose(ry, torch.tensor(0.0))
    assert torch.allclose(rz, torch.tensor(1.0))


def test_constitutive_zero_velocity():
    # If velocity is zero, deviatoric stress should be zero.
    model = SpatialPINN(spatial_dim=3)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0)

    x = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

    # 3D Constitutive
    # out: [vx, vy, vz, sxx, syy, szz, sxy, syz, sxz]
    # Indices 3,4,5,6,7,8 are stresses.
    # If we set biases to 0, stresses are 0.
    # If we set weights to 0, derivatives of velocity are 0.

    r_xx, r_yy, r_zz, r_xy, r_yz, r_xz, r_vol = Physics.constitutive_3d(
        model, x, eta=1e21, S0=1e8, V0=1e-9
    )

    assert torch.allclose(r_xx, torch.tensor(0.0))
    assert torch.allclose(r_vol, torch.tensor(0.0))


def test_traction_free_surface():
    # At z=-1, sig_xz, sig_yz, sig_zz should be 0.
    model = SpatialPINN(spatial_dim=3)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0)  # Biases=0 -> stress=0

    x_surf = torch.tensor([[0.5, 0.5, -1.0]])
    rxz, ryz, rzz = Physics.traction_free_surface(model, x_surf, S0=1e8)

    assert torch.allclose(rxz, torch.tensor(0.0))
    assert torch.allclose(ryz, torch.tensor(0.0))
    assert torch.allclose(rzz, torch.tensor(0.0))
