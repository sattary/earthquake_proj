import torch
from src.core.model import SpatialPINN


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
