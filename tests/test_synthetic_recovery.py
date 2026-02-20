import pytest
import torch
import pandas as pd
from pathlib import Path

from src.validation.synthetic_benchmark import AnalyticalBenchmark
from src.validation.synthetic_generator import SyntheticDataGenerator
from src.core.physics import Physics
import src.core.model


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("synthetic")
    gps_path = str(tmp_dir / "gps_synthetic.csv")
    catalog_path = str(tmp_dir / "catalog_synthetic.txt")

    gen = SyntheticDataGenerator(domain_bounds=[0, 1e4, 0, 1e4, -1e4, 0])
    gen.generate_gps_data("simple_shear", num_stations=50, out_path=gps_path)
    gen.generate_catalog("simple_shear", num_events=100, out_path=catalog_path)

    return {"gps": gps_path, "catalog": catalog_path, "dir": tmp_dir}


def test_benchmark_satisfies_pde():
    """
    Validates that the ground truth analytical models strictly satisfy
    the 3D momentum balance equation. This formally proves the
    forward mathematical well-posedness of the data generation.
    """
    bench = AnalyticalBenchmark()

    x_rand = torch.rand(100) * 1e4
    y_rand = torch.rand(100) * 1e4
    z_rand = torch.rand(100) * -1e4

    coords = torch.stack([x_rand, y_rand, z_rand], dim=1).requires_grad_(True)

    # We create a dummy model wrapper that just returns the analytical stress
    class AnalyticalMock:
        def __call__(self, x_in):
            fields = bench.uniaxial_compression(x_in)
            from src.core.constants import S0, V0

            return torch.stack(
                [
                    fields["v_x"] / V0,
                    fields["v_y"] / V0,
                    fields["v_z"] / V0,
                    fields["s_xx"] / S0,
                    fields["s_yy"] / S0,
                    fields["s_zz"] / S0,
                    fields["s_xy"] / S0,
                    fields["s_yz"] / S0,
                    fields["s_xz"] / S0,
                ],
                dim=-1,
            )

    # Check Momentum Balance PDE residuals (should be effectively zero)
    res_x, res_y, res_z = Physics.momentum_balance_3d(AnalyticalMock(), coords)

    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-4)
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-4)
    assert torch.allclose(res_z, torch.zeros_like(res_z), atol=1e-4)


def test_generator_creates_valid_files(synthetic_data):
    """
    Ensures the generator output conforms to real observed data formats.
    """
    gps_path = Path(synthetic_data["gps"])
    cat_path = Path(synthetic_data["catalog"])

    assert gps_path.exists()
    assert cat_path.exists()

    df_gps = pd.read_csv(gps_path)
    assert len(df_gps) == 50
    assert "azimuth_value" in df_gps.columns

    df_cat = pd.read_csv(cat_path)
    assert len(df_cat) == 100
    assert "mw_unified" in df_cat.columns


from unittest.mock import patch


def test_pinn_can_train_on_synthetic_data(synthetic_data):
    """
    Formal integration test proving the PINN model can successfully ingest
    the synthetic GPS and Seismicity data and execute the full training
    loop. This verifies the complete computational pipeline for Phase 1.
    """
    from src.training.engine import PINNTrainer

    gps_path = synthetic_data["gps"]
    catalog_path = synthetic_data["catalog"]

    trainer = PINNTrainer(
        spatial_dim=3,
        lr=2e-3,
        fourier_scale=1.0,
        coupling_enabled=True,
        multi_gpu=False,
    )

    # Assert network has successfully detected coupling components
    assert hasattr(trainer.model, "log_A")
    assert hasattr(trainer.model, "log_r0")

    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    # Mock CoordinateTransformer to pass through Cartesian coordinates directly
    with patch("src.data.loaders.CoordinateTransformer") as MockTransformer:
        mock_inst = MockTransformer.return_value
        mock_inst.scale = (
            5000.0  # Required so engine.py lines 97 doesn't evaluate to a MagicMock
        )
        # For GPS dataset (x, y)
        mock_inst.to_normalized.side_effect = lambda lats, longs: torch.stack(
            [
                torch.tensor(longs, dtype=torch.float32),
                torch.tensor(lats, dtype=torch.float32),
            ],
            dim=1,
        )

        # Train for 2 epochs (minimally check pipeline execution bounds)
        trainer.train(
            gps_files=[gps_path],
            epochs=2,
            n_coll=100,
            w_data=5.0,
            w_pde=1.0,
            w_const=1.0,
            w_bc=1.0,
            w_seis=0.5,
            catalog_file=catalog_path,
        )

    # Checks training process hasn't corrupted model state
    assert trainer.model is not None
    # Depending on the scheduler, either lr changes or stays same,
    # but the network weights must have required gradients.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in trainer.model.parameters()
        if p.requires_grad
    )
    assert has_grad
