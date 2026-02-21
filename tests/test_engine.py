import os
from src.training.engine import PINNTrainer
from src.core.model import SpatialPINN


def test_trainer_initialization():
    trainer = PINNTrainer(spatial_dim=2)
    assert isinstance(trainer.model, SpatialPINN)
    assert trainer.device.type in ["cuda", "cpu"]


def test_trainer_train_step_smoke(tmp_path):
    # This is a smoke test to ensure one epoch doesn't crash
    # Needs a mock GPS file
    import pandas as pd

    df = pd.DataFrame(
        {
            "latitude": [35.5, 36.0],
            "longitude": [51.5, 52.0],
            "azimuth_value": [10.0, 20.0],
        }
    )
    gps_file = tmp_path / "smoke_gps.csv"
    df.to_csv(gps_file, index=False)

    trainer = PINNTrainer(spatial_dim=2)
    # Patch parameters for speed
    trainer.train(gps_files=[str(gps_file)], epochs=2, n_coll=100, velocity_file=None)

    assert os.path.exists("checkpoints/final_model.pth")
    assert os.path.exists("results/tables/training_history.json")
