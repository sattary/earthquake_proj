import json
from pathlib import Path
from tqdm import tqdm
from src.analysis.synthetic_eval import evaluate_synthetic_recovery


def run_robustness_sweeps(
    regime: str = "simple_shear",
    epochs: int = 200,
    out_file: str = "results/tables/robustness.json",
):
    """
    Automates the evaluation of the PINN against varying levels of
    GPS station sparsity and GPS measurement noise.
    """
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {"sparsity_sweep": [], "noise_sweep": []}

    # Sweep 1: Sparsity (Varying GPS count, Fixed Noise)
    # How few stations do we need before the PINN fails to resolve the stress tensor?
    gps_counts = [10, 50, 100, 200, 500]
    fixed_noise = 2.0

    print("\n--- Sweeping Data Sparsity ---")
    for count in tqdm(gps_counts, desc="Sparsity"):
        res = evaluate_synthetic_recovery(
            regime=regime, num_gps=count, noise_std=fixed_noise, epochs=epochs
        )
        results["sparsity_sweep"].append(res)

        # Save intermediate
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

    # Sweep 2: Noise Robustness (Varying Noise, Fixed Dense GPS)
    # How much Gaussian error can the PDE regularize away?
    noise_levels = [0.0, 1.0, 5.0, 10.0, 20.0]
    fixed_count = 500

    print("\n--- Sweeping Measurement Noise ---")
    for noise in tqdm(noise_levels, desc="Noise"):
        res = evaluate_synthetic_recovery(
            regime=regime, num_gps=fixed_count, noise_std=noise, epochs=epochs
        )
        results["noise_sweep"].append(res)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"✅ Robustness sweeps complete. Saved to {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    run_robustness_sweeps(epochs=args.epochs)
