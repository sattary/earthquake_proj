import json
from pathlib import Path
from src.analysis.synthetic_eval import evaluate_synthetic_recovery


def run_physical_ablation(
    regime: str = "simple_shear",
    epochs: int = 200,
    out_file: str = "results/tables/ablation_seismicity.json",
):
    """
    Runs a systematic ablation on the PINN by toggling Rate-and-State friction.
    Proves that `w_seis` coupling is required to resolve the absolute magnitude ambiguity
    that normally plagues GPS-only boundary inversions.
    """
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    print("\n=============================================")
    print("   PINN PHYSICS ABLATION: RATE-AND-STATE    ")
    print("=============================================\n")

    # 1. Uncoupled Inversion (PDE + GPS only)
    # The PINN can solve for the relative tensor shape, but the absolute magnitude
    # will be wildly unconstrained, resulting in massive L2 errors against the Truth.
    print("-> Traning UNCOUPLED Inversion (w_seis = 0.0)")
    uncoupled_res = evaluate_synthetic_recovery(
        regime=regime, num_gps=500, noise_std=2.0, epochs=epochs, w_seis=0.0
    )
    results["uncoupled"] = uncoupled_res

    # 2. Fully Coupled Inversion (PDE + GPS + Seismicity)
    # The rate-and-state equation bounds the absolute Coulomb stress magnitude,
    # pulling the L2 error down significantly.
    print("\n-> Traning FULLY COUPLED Inversion (w_seis = 1.0)")
    coupled_res = evaluate_synthetic_recovery(
        regime=regime, num_gps=500, noise_std=2.0, epochs=epochs, w_seis=1.0
    )
    results["coupled"] = coupled_res

    # Dump to JSON for plotting
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Ablation Study Complete.")
    print(f"Uncoupled L2 Error : {uncoupled_res['l2_error'] * 100:.2f}%")
    print(f"Coupled   L2 Error : {coupled_res['l2_error'] * 100:.2f}%")
    print(f"Saved logs to {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    run_physical_ablation(epochs=args.epochs)
