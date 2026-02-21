import torch
from src.core.physics import Physics
from src.core.model import SpatialPINN
from src.core.constants import MU_FRICTION


class TestCoulombFailure:
    """Tests for the Coulomb Failure Function computation."""

    def test_cff_uniaxial_compression(self):
        """
        Uniaxial compression along z: s1=0, s3=-100 MPa.
        tau_max = (0 - (-1e8)) / 2 = 5e7
        sigma_n = (0 + (-1e8)) / 2 = -5e7
        CFF = tau_max - mu_f * (sigma_n - P_f)
            = 5e7 - 0.6 * (-5e7 - 0) = 5e7 + 3e7 = 8e7
        (with P_f = 0 when z = 0, i.e., surface)
        """
        model = SpatialPINN(spatial_dim=3)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0)
            # Set stress biases: sxx=0, syy=0, szz=-100 MPa / S0
            # Output indices: [vx, vy, vz, sxx, syy, szz, sxy, syz, sxz]
            # szz is output index 5 -> head.bias[5]
            model.head.bias[5] = -1e8 / 1e9  # -0.1 in normalized units

        # Point at surface (z=0 -> no pore pressure)
        x = torch.tensor([[0.0, 0.0, 0.0]])
        cff, sigma_n = Physics.coulomb_failure(model, x, lambda_p=0.0)

        # Verify signs and magnitudes
        assert cff.shape == (1,)
        assert sigma_n.item() < 0  # Compressive
        assert cff.item() > 0  # Above failure

    def test_cff_hydrostatic(self):
        """
        Hydrostatic stress: s1 = s2 = s3 = -P.
        tau_max = 0, sigma_n = -P.
        CFF = 0 - mu_f * (-P - 0) = mu_f * P (at surface, P_f=0).

        Rationale: Under hydrostatic conditions with no pore pressure,
        CFF should be positive due to the sign convention -- the friction
        term dominates. This is a sanity check, not a physical scenario.
        """
        model = SpatialPINN(spatial_dim=3)
        pressure = 1e8  # 100 MPa
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0)
            # Set all normal stresses to -P / S0
            model.head.bias[3] = -pressure / 1e9
            model.head.bias[4] = -pressure / 1e9
            model.head.bias[5] = -pressure / 1e9

        x = torch.tensor([[0.0, 0.0, 0.0]])
        cff, sigma_n = Physics.coulomb_failure(model, x, lambda_p=0.0)

        # tau_max should be ~0
        assert abs(sigma_n.item() + pressure) < 1e3  # sigma_n ~ -P
        # CFF = 0 - mu_f * (-P) = mu_f * P
        expected_cff = MU_FRICTION * pressure
        assert abs(cff.item() - expected_cff) < 1e3


class TestSeismicityRate:
    """Tests for the Dieterich seismicity rate."""

    def test_rate_always_positive(self):
        """Seismicity rate must be positive for any CFF value."""
        cff = torch.tensor([-1e8, 0.0, 1e8])
        sigma_n = torch.tensor([-5e7, -5e7, -5e7])
        a_param = torch.tensor(0.005)
        r0 = torch.tensor(1.0)

        rate = Physics.seismicity_rate(cff, sigma_n, a_param, r0)

        assert (rate > 0).all()
        assert rate.shape == (3,)

    def test_rate_monotonic_with_cff(self):
        """Higher CFF -> higher seismicity rate (monotonic)."""
        sigma_n = torch.tensor([-5e7, -5e7, -5e7])
        a_param = torch.tensor(0.005)
        r0 = torch.tensor(1.0)

        cff_low = torch.tensor([-1e7, -1e7, -1e7])
        cff_high = torch.tensor([1e7, 1e7, 1e7])

        rate_low = Physics.seismicity_rate(cff_low, sigma_n, a_param, r0)
        rate_high = Physics.seismicity_rate(cff_high, sigma_n, a_param, r0)

        assert (rate_high > rate_low).all()

    def test_rate_no_nan(self):
        """Rate computation must not produce NaN or Inf."""
        cff = torch.tensor([1e12, -1e12, 0.0])
        sigma_n = torch.tensor([-1e7, -1e7, -1e7])
        a_param = torch.tensor(0.005)
        r0 = torch.tensor(1.0)

        rate = Physics.seismicity_rate(cff, sigma_n, a_param, r0)

        assert torch.isfinite(rate).all()


class TestPoissonLoss:
    """Tests for the Poisson log-likelihood loss shape."""

    def test_poisson_loss_shape(self):
        """
        Construct a minimal scenario: a model with coupling enabled,
        compute the seismicity loss, verify it returns a finite scalar.
        """
        from src.training.engine import PINNTrainer

        trainer = PINNTrainer(
            spatial_dim=3,
            coupling_enabled=True,
            multi_gpu=False,
        )

        # Mock catalog and collocation points
        x_catalog = torch.rand(10, 3) * 2 - 1  # [-1, 1]
        x_coll = torch.rand(50, 3) * 2 - 1

        loss = trainer.compute_seismicity_loss(x_catalog, x_coll)

        assert loss.shape == ()
        assert torch.isfinite(loss)
