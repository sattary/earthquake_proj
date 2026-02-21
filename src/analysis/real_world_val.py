import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.core.model import SpatialPINN
from src.data.transformers import CoordinateTransformer


def evaluate_against_focal_mechanisms(
    model_path: str = "checkpoints/final_model.pth",
    focal_mech_file: str = "data/kinematic_data/stress_SS.csv",
    out_file: str = "results/tables/focal_errors.json",
):
    """
    Validates a trained PINN model against independent real-world
    focal mechanism inversions (e.g. World Stress Map data).
    Outputs the angular deviation histogram.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    # The focal mechanism CSV should have ['longitude', 'latitude', 'azimuth_value']
    # representing the S_Hmax (maximum horizontal compressive stress) azimuth
    try:
        df_fm = pd.read_csv(focal_mech_file)
    except FileNotFoundError:
        print(f"File {focal_mech_file} not found. Skipping real-world validation.")
        return

    lons = df_fm["longitude"].values
    lats = df_fm["latitude"].values
    true_azimuths = df_fm["azimuth_value"].values

    # Standardize to 10km depth for map comparisons
    depths = np.full_like(lons, 10000.0)

    # 2. Setup PINN
    transformer = CoordinateTransformer(lats, lons)

    coords = transformer.to_normalized(lats, lons)

    # Map [0, 30km] to [1, -1]
    z_norm = (depths / 15000.0) - 1.0

    x_input = torch.stack(
        [coords[:, 0], coords[:, 1], torch.tensor(z_norm, dtype=torch.float32)], dim=1
    ).to(device)

    # Note: SpatialPINN defaults to spatial_dim=3
    model = SpatialPINN(spatial_dim=3, coupling_enabled=True)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Checkpoint {model_path} not found. Train the model first.")
        return

    model.to(device)
    model.eval()

    # 3. Predict Stress Tensor
    with torch.no_grad():
        out = model(x_input)
        s_xx = out[:, 3]
        s_yy = out[:, 4]
        s_xy = out[:, 6]

    # S_Hmax predicting (CCW from East in Math logic)
    theta_rad_math = 0.5 * torch.atan2(2 * s_xy, s_xx - s_yy)

    # Convert math angle to geographic azimuth (Clockwise from North)
    theta_rad_geo = (torch.pi / 2) - theta_rad_math
    pred_azimuths = torch.rad2deg(theta_rad_geo).cpu().numpy()

    # Wrap to [0, 180) degrees
    pred_azimuths = pred_azimuths % 180.0
    true_azimuths = true_azimuths % 180.0

    # 4. Compute Angular Deviations
    # Shortest circular distance between two lines (modulo 180)
    diffs = np.abs(pred_azimuths - true_azimuths)
    angular_errors = np.minimum(diffs, 180.0 - diffs)

    mae = np.mean(angular_errors)
    std = np.std(angular_errors)

    print("\n--- Focal Mechanism Validation ---")
    print(f"Mean Angular Error : {mae:.2f}° ± {std:.2f}°")
    print(f"Max Error          : {np.max(angular_errors):.2f}°")

    # Save the full histogram distribution
    results = {"mae": float(mae), "std": float(std), "errors": angular_errors.tolist()}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Saved validation results to", out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="checkpoints/final_model.pth")
    parser.add_argument(
        "--focal-mech-file", default="data/kinematic_data/stress_SS.csv"
    )
    args = parser.parse_args()

    evaluate_against_focal_mechanisms(
        model_path=args.model_path, focal_mech_file=args.focal_mech_file
    )
