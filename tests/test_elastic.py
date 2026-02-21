import torch
from src.core.physics import Physics
from src.core.model import SpatialPINN
from src.core.constants import S0, V0, MU_BASELINE


class TestElasticConstitutive:
    """Tests for the elastic (Hooke's law) constitutive residuals."""

    def test_zero_displacement_zero_stress(self):
        """
        If both displacement and stress fields are zero (all weights/biases = 0),
        all constitutive residuals should be zero.

        Rationale: Hooke's law with zero strain gives zero stress. If the
        network also predicts zero stress, the residual is exactly zero.
        """
        model = SpatialPINN(spatial_dim=3)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0)

        x = torch.rand(10, 3, requires_grad=True)
        results = Physics.elastic_constitutive_3d(
            model,
            x,
            lam=MU_BASELINE,
            mu=MU_BASELINE,
            stress_scale=S0,
            disp_scale=V0,
        )

        for r in results:
            assert r.shape == (10,)
            # All residuals should be zero because both sides of Hooke's law are zero
            assert torch.allclose(r, torch.zeros(10), atol=1e-6)

    def test_return_shape(self):
        """
        Elastic constitutive returns a tuple of 7 tensors, each (N,).
        """
        model = SpatialPINN(spatial_dim=3)
        x = torch.rand(5, 3, requires_grad=True)
        results = Physics.elastic_constitutive_3d(
            model,
            x,
            lam=30e9,
            mu=30e9,
        )

        assert len(results) == 7
        for r in results:
            assert r.shape == (5,)

    def test_volume_residual_is_zero(self):
        """
        The elastic law does not impose incompressibility, so
        the volumetric residual should always be zero.
        """
        model = SpatialPINN(spatial_dim=3)
        x = torch.rand(8, 3, requires_grad=True)
        results = Physics.elastic_constitutive_3d(
            model,
            x,
            lam=30e9,
            mu=30e9,
        )
        r_vol = results[6]
        assert torch.allclose(r_vol, torch.zeros_like(r_vol), atol=1e-10)

    def test_gradient_flow(self):
        """
        The elastic constitutive residuals must propagate gradients
        back to the model parameters (required for PINN training).
        """
        model = SpatialPINN(spatial_dim=3)
        x = torch.rand(4, 3, requires_grad=True)

        results = Physics.elastic_constitutive_3d(
            model,
            x,
            lam=30e9,
            mu=30e9,
        )
        loss = sum(torch.mean(r**2) for r in results)
        loss.backward()

        # At least one model parameter should have a non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "Gradients must flow through elastic constitutive law"


class TestLearnableParameters:
    """Tests for the learnable coupling parameters in SpatialPINN."""

    def test_coupling_disabled_by_default(self):
        """By default, coupling is disabled and parameters are absent."""
        model = SpatialPINN(spatial_dim=3)
        assert model.coupling_enabled is False
        assert not hasattr(model, "log_A")

    def test_coupling_enabled_creates_parameters(self):
        """When coupling is enabled, learnable parameters are created."""
        model = SpatialPINN(spatial_dim=3, coupling_enabled=True)
        assert model.coupling_enabled is True

        # Parameters exist and are nn.Parameter
        assert isinstance(model.log_A, torch.nn.Parameter)
        assert isinstance(model._lambda_p, torch.nn.Parameter)
        assert isinstance(model.log_r0, torch.nn.Parameter)

    def test_a_param_positive(self):
        """a_param property always returns a positive value."""
        model = SpatialPINN(spatial_dim=3, coupling_enabled=True)
        assert model.a_param.item() > 0

    def test_pore_pressure_ratio_clamped(self):
        """pore_pressure_ratio is clamped to [0, 1]."""
        model = SpatialPINN(spatial_dim=3, coupling_enabled=True)
        with torch.no_grad():
            model._lambda_p.fill_(5.0)
        assert model.pore_pressure_ratio.item() == 1.0

        with torch.no_grad():
            model._lambda_p.fill_(-2.0)
        assert model.pore_pressure_ratio.item() == 0.0

    def test_parameters_in_optimizer(self):
        """Coupling parameters must appear in model.parameters()."""
        model = SpatialPINN(spatial_dim=3, coupling_enabled=True)
        param_names = [n for n, _ in model.named_parameters()]
        assert "log_A" in param_names
        assert "_lambda_p" in param_names
        assert "log_r0" in param_names

    def test_backward_compatibility(self):
        """
        Models without coupling should produce the same output shape
        and not break when loaded from old checkpoints.
        """
        model_old = SpatialPINN(spatial_dim=3, coupling_enabled=False)
        model_new = SpatialPINN(spatial_dim=3, coupling_enabled=True)

        # Old checkpoint (no coupling params)
        old_state = model_old.state_dict()

        # Load with strict=False should work
        missing, unexpected = model_new.load_state_dict(old_state, strict=False)
        assert len(unexpected) == 0
        # Missing keys should be exactly the coupling params
        coupling_keys = {"log_A", "_lambda_p", "log_r0"}
        assert coupling_keys == set(missing)
