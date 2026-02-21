import torch
from typing import Dict
from pathlib import Path
from unittest.mock import patch

from src.training.engine import PINNTrainer
from src.core.constants import S0
from src.validation.synthetic_generator import SyntheticDataGenerator


def evaluate_synthetic_recovery(
    regime: str = "simple_shear",
    num_gps: int = 500,
    num_events: int = 2000,
    noise_std: float = 2.0,
    epochs: int = 200,
    w_seis: float = 1.0,
) -> Dict[str, float]:
    """
    Trains a PINN on synthetic data and computes relative L2 error
    against the analytical ground truth stress tensor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_dir = Path("data/synthetic_eval")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Dataset
    generator = SyntheticDataGenerator()
    gps_path = str(tmp_dir / f"gps_{regime}.csv")
    cat_path = str(tmp_dir / f"cat_{regime}.csv")
    truth_path = str(tmp_dir / f"truth_{regime}.pt")

    generator.generate_gps_data(regime, num_gps, noise_std, gps_path)
    generator.generate_catalog(regime, num_events, out_path=cat_path)
    truth_dict = generator.generate_ground_truth_grid(
        regime, resolution=20, out_path=truth_path
    )

    # 2. Train Model (Mocking Projection to use Cartesian Synthetics Directly)
    # The analytical bounds are (0, 100km, 0, 100km)
    with patch("src.data.loaders.CoordinateTransformer") as MockTransformer:
        mock_inst = MockTransformer.return_value
        mock_inst.scale = 50000.0  # Half-width of 100km domain
        mock_inst.min_x, mock_inst.max_x = 0.0, 1e5
        mock_inst.min_y, mock_inst.max_y = 0.0, 1e5

        # Bypass geographic projection, pass raw Cartesian (x, y)
        mock_inst.to_normalized.side_effect = lambda lats, longs: torch.stack(
            [
                torch.tensor(longs, dtype=torch.float32),
                torch.tensor(lats, dtype=torch.float32),
            ],
            dim=1,
        )

        trainer = PINNTrainer(
            spatial_dim=3,
            lr=1e-3,
            checkpoint_dir="checkpoints/eval",
            multi_gpu=False,
            coupling_enabled=True if w_seis > 0 else False,
        )

        trainer.train(
            gps_files=[gps_path],
            epochs=epochs,
            n_coll=2000,  # Dense physics sampling
            w_data=5.0,
            w_pde=1.0,
            w_const=1.0,
            w_bc=1.0,
            w_seis=w_seis,
            catalog_file=cat_path,
        )

    # 3. Predict on Ground Truth Grid
    model = trainer.model.eval()

    coords = truth_dict["coords"].to(device)  # Shape [N, 3] MKS
    # Normalize coords for NN input
    x_norm = (coords[:, 0] / mock_inst.scale) - 1.0
    y_norm = (coords[:, 1] / mock_inst.scale) - 1.0
    z_norm = (coords[:, 2] / 15000.0) + 1.0  # 0 to -30km -> -1 to 1
    coords_norm = torch.stack([x_norm, y_norm, z_norm], dim=1)

    with torch.no_grad():
        out = model(coords_norm)
        # out[:, 0-2] are velocities, [:, 3-8] are stresses (xx, yy, zz, xy, yz, xz)
        pred_stress = out[:, 3:9] * S0

    # 4. Compute Relative L2 Error
    gt_stress = torch.stack(
        [
            truth_dict["s_xx"].flatten(),
            truth_dict["s_yy"].flatten(),
            truth_dict["s_zz"].flatten(),
            truth_dict["s_xy"].flatten(),
            truth_dict["s_yz"].flatten(),
            truth_dict["s_xz"].flatten(),
        ],
        dim=1,
    ).to(device)

    # Relative L2 Norm: ||S_pred - S_gt||_2 / ||S_gt||_2
    diff_norm = torch.linalg.matrix_norm(pred_stress - gt_stress, ord="fro")
    gt_norm = torch.linalg.matrix_norm(gt_stress, ord="fro")

    rel_error = (diff_norm / gt_norm).item()

    print("\n--- Synthetic Evaluation Result ---")
    print(f"Regime: {regime}")
    print(f"Stations: {num_gps}, Noise: {noise_std}°")
    print(f"L2 Stress Error: {rel_error * 100:.2f}%\n")

    return {
        "regime": regime,
        "num_gps": num_gps,
        "noise_std": noise_std,
        "l2_error": rel_error,
        "val_loss": trainer.history["loss"][-1],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default="simple_shear")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=2.0)
    args = parser.parse_args()

    evaluate_synthetic_recovery(
        regime=args.regime, epochs=args.epochs, noise_std=args.noise_std
    )
