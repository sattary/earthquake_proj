"""
CLI for Earthquake PINN Project.

Supports both hierarchical config files (YAML/JSON) and flat CLI options.
"""

import typer
import glob
import os
from typing import Optional

from src.training.engine import PINNTrainer
from src.training.tuner import run_tuning

from src.core.config import load_train_config, save_train_config, TrainConfig
from src.git_automation import add_auto_push_args, create_auto_push_callback
from src.training.multi_gpu import detect_kaggle_multi_gpu

app = typer.Typer(help="L3 Earthquake PINN Operational Pipeline")


def _resolve_config(
    config_file: Optional[str],
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    **overrides,
) -> TrainConfig:
    """
    Resolve final config from file and CLI overrides.

    CLI options take precedence over config file values.
    """
    if config_file:
        cfg = load_train_config(config_file)
    else:
        cfg = TrainConfig()

    for key, value in overrides.items():
        if value is not None:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            elif "." in key:
                parts = key.split(".")
                obj = cfg
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

    if epochs is not None:
        cfg.optim.epochs = epochs
    if lr is not None:
        cfg.optim.lr = lr

    return cfg


@app.command()
@add_auto_push_args()
def train(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (YAML or JSON). CLI options override config values.",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Number of training epochs (overrides config)",
    ),
    lr: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Learning rate (overrides config)",
    ),
    n_coll: Optional[int] = typer.Option(
        None,
        "--n-coll",
        help="Number of physics collocation points (overrides config)",
    ),
    w_data: Optional[float] = typer.Option(
        None,
        "--w-data",
        help="Weight for Data Loss (overrides config)",
    ),
    w_pde: Optional[float] = typer.Option(
        None,
        "--w-pde",
        help="Weight for PDE Loss (overrides config)",
    ),
    w_const: Optional[float] = typer.Option(
        None,
        "--w-const",
        help="Weight for Constitutive Law Loss (overrides config)",
    ),
    w_bc: Optional[float] = typer.Option(
        None,
        "--w-bc",
        help="Weight for Boundary Condition Loss (overrides config)",
    ),
    w_seis: Optional[float] = typer.Option(
        None,
        "--w-seis",
        help="Weight for Seismicity Coupling Loss (overrides config)",
    ),
    fourier_scale: Optional[float] = typer.Option(
        None,
        "--fourier-scale",
        help="Fourier Feature scale (overrides config)",
    ),
    spatial_dim: Optional[int] = typer.Option(
        None,
        "--spatial-dim",
        help="Spatial Dimension (2 or 3) (overrides config)",
    ),
    velocity_file: Optional[str] = typer.Option(
        None,
        "--velocity-file",
        help="Path to Velocity Model (overrides config)",
    ),
    catalog_file: Optional[str] = typer.Option(
        None,
        "--catalog-file",
        help="Path to earthquake catalog for seismicity coupling (overrides config)",
    ),
    constitutive: Optional[str] = typer.Option(
        None,
        "--constitutive",
        help="Constitutive law: 'viscous' or 'elastic' (overrides config)",
    ),
    resume: bool = typer.Option(False, help="Resume from latest checkpoint"),
    multi_gpu: Optional[bool] = typer.Option(
        None,
        "--multi-gpu/--no-multi-gpu",
        help="Use multiple GPUs if available (overrides config)",
    ),
    save_config: Optional[str] = typer.Option(
        None,
        "--save-config",
        help="Save the resolved config to this path after training starts",
    ),
    auto_push_interval: Optional[int] = typer.Option(
        None,
        "--auto-push-interval",
        help="Interval of epochs to push artifacts to git",
    ),
    auto_push_dry_run: bool = typer.Option(
        False,
        "--auto-push-dry-run",
        help="Run auto-push in dry run mode (no actual pushes)",
    ),
    force_auto_push: bool = typer.Option(
        False,
        "--force-auto-push",
        help="Force auto-push even if not in cloud environment",
    ),
    auto_push_pat: Optional[str] = typer.Option(
        None,
        "--auto-push-pat",
        help="GitHub Personal Access Token for auto-push (can also use GITHUB_PAT env var)",
    ),
):
    """
    Train the PINN model.

    Examples:

    \b
    # Use config file only
    earthquake-pinn train --config configs/default.yaml

    \b
    # Use config with CLI overrides
    earthquake-pinn train --config configs/default.yaml --epochs 5000 --lr 1e-4

    \b
    # Use CLI only (no config file)
    earthquake-pinn train --epochs 10000 --spatial-dim 3
    """
    overrides = {
        "n_coll": n_coll,
        "w_data": w_data,
        "w_pde": w_pde,
        "w_const": w_const,
        "w_bc": w_bc,
        "w_seis": w_seis,
        "fourier_scale": fourier_scale,
        "spatial_dim": spatial_dim,
        "velocity_file": velocity_file,
        "catalog_file": catalog_file,
        "constitutive": constitutive,
        "multi_gpu": multi_gpu,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    cfg = _resolve_config(config, epochs, lr, **overrides)

    print("=" * 60)
    print("Training Configuration:")
    print("-" * 60)
    print(
        f"  Model: spatial_dim={cfg.model.spatial_dim}, fourier_scale={cfg.model.fourier_scale}"
    )
    print(f"  Optim: epochs={cfg.optim.epochs}, lr={cfg.optim.lr}")
    print(
        f"  Loss:  w_data={cfg.loss.w_data}, w_pde={cfg.loss.w_pde}, w_seis={cfg.loss.w_seis}"
    )
    print(
        f"  Physics: constitutive={cfg.physics.constitutive}, coupling={cfg.physics.coupling_enabled}"
    )
    print(f"  Data:   velocity={cfg.data.velocity_file}")
    if cfg.data.catalog_file:
        print(f"  Catalog: {cfg.data.catalog_file}")
    print("=" * 60)

    if save_config:
        save_train_config(cfg, save_config)
        print(f"Config saved to: {save_config}")

    auto_push_cb = create_auto_push_callback(
        run_dir="runs/current",  # The engine outputs to results/, but zip_packer will zip based on run_dir. Need a logic here if results dir rotates. Or just stick to "."
        interval=auto_push_interval,
        dry_run=auto_push_dry_run,
        force=force_auto_push,
        pat=auto_push_pat,
        include_checkpoints=True,
    )
    if auto_push_cb:
        # Override the run_dir to point where checkpoints and history are saved
        # Actually in engine.py: self.checkpoint_dir = Path(checkpoint_dir) which defaults to "checkpoints"
        # and history is "results/tables/training_history.json"
        # Ideally, zip everything in the root or a dedicated run directory. We will zip the whole repo if "."
        # But we'll set run_dir to "." to capture everything.
        auto_push_cb.run_dir = os.path.abspath(".")
        auto_push_cb.packer.run_dir = os.path.abspath(".")

    coupling_on = cfg.loss.w_seis > 0.0 and cfg.data.catalog_file is not None
    cfg.physics.coupling_enabled = coupling_on

    if cfg.multi_gpu is None and detect_kaggle_multi_gpu():
        print("[Auto-Detect] Kaggle multi-GPU detected. Enabling multi-GPU mode.")
        cfg.multi_gpu = True

    trainer = PINNTrainer(
        spatial_dim=cfg.model.spatial_dim,
        lr=cfg.optim.lr,
        fourier_scale=cfg.model.fourier_scale,
        auto_push_callback=auto_push_cb,
        multi_gpu=cfg.multi_gpu,
        constitutive=cfg.physics.constitutive,
        coupling_enabled=coupling_on,
    )

    gps_files = glob.glob(f"{cfg.data.data_dir}/{cfg.data.gps_pattern}")
    if not gps_files:
        gps_files = glob.glob("data/kinematic_data/gps_strain_*.csv")

    if not gps_files:
        print("Error: No GPS files found.")
        raise typer.Exit(code=1)

    resume_path = None
    if resume:
        checkpoints = glob.glob("checkpoints/checkpoint_epoch_*.pth")
        if checkpoints:

            def extract_epoch(p):
                try:
                    return int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
                except Exception:
                    return -1

            latest_checkpoint = max(checkpoints, key=extract_epoch)
            resume_path = latest_checkpoint
            print(f"Resuming from: {resume_path}")

    trainer.train(
        gps_files,
        epochs=cfg.optim.epochs,
        n_coll=cfg.optim.n_coll,
        w_data=cfg.loss.w_data,
        w_pde=cfg.loss.w_pde,
        w_const=cfg.loss.w_const,
        w_bc=cfg.loss.w_bc,
        w_seis=cfg.loss.w_seis,
        velocity_file=cfg.data.velocity_file,
        catalog_file=cfg.data.catalog_file,
        resume_from_checkpoint=resume_path,
    )


@app.command()
def tune(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (YAML or JSON)",
    ),
    trials: Optional[int] = typer.Option(
        None,
        "--trials",
        help="Number of Optuna trials (overrides config)",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Epochs per trial (overrides config)",
    ),
    spatial_dim: Optional[int] = typer.Option(
        None,
        "--spatial-dim",
        help="Spatial Dimension (overrides config)",
    ),
    velocity_file: Optional[str] = typer.Option(
        None,
        "--velocity-file",
        help="Path to Velocity Model (overrides config)",
    ),
    multi_gpu: Optional[bool] = typer.Option(
        None,
        "--multi-gpu/--no-multi-gpu",
        help="Use multiple GPUs for tuning trials (overrides config)",
    ),
    constitutive: Optional[str] = typer.Option(
        None,
        "--constitutive",
        help="Constitutive law: 'viscous' or 'elastic' (overrides config)",
    ),
    auto_push: bool = typer.Option(
        False,
        "--auto-push",
        help="Enable async push to GitHub after tuning",
    ),
    train_after: bool = typer.Option(
        False,
        "--train-after",
        help="Run main training with best config immediately after tuning",
    ),
):
    """
    Run Hyperparameter Tuning.

    Examples:

    \b
    # Use config file
    earthquake-pinn tune --config configs/tuning.yaml

    \b
    # Use CLI options
    earthquake-pinn tune --trials 50 --epochs 100 --spatial-dim 3
    """
    overrides = {
        "spatial_dim": spatial_dim,
        "velocity_file": velocity_file,
        "multi_gpu": multi_gpu,
        "constitutive": constitutive,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    cfg = _resolve_config(config, epochs=epochs, **overrides)

    if cfg.multi_gpu is None and detect_kaggle_multi_gpu():
        print("[Auto-Detect] Kaggle multi-GPU detected. Enabling multi-GPU mode.")
        cfg.multi_gpu = True

    auto_push_cb = None
    if auto_push:
        auto_push_cb = create_auto_push_callback(
            run_dir=".",
            interval=1000000,
            dry_run=False,
            force=False,
            pat=None,
            include_checkpoints=False,
        )

    run_tuning(
        n_trials=trials or 20,
        epochs=epochs or cfg.optim.epochs,
        spatial_dim=cfg.model.spatial_dim,
        velocity_file=cfg.data.velocity_file,
        multi_gpu=cfg.multi_gpu,
        constitutive=cfg.physics.constitutive,
        coupling_enabled=cfg.physics.coupling_enabled,
        base_cfg=cfg,
        auto_push_callback=auto_push_cb,
    )

    if train_after:
        import subprocess

        print("\n[Auto-Train] Launching main training with best Optuna config...")
        cmd = [
            "uv",
            "run",
            "earthquake-proj",
            "train",
            "--config",
            "results/optuna/best_config.yaml",
        ]
        if auto_push:
            cmd.append("--auto-push")
        subprocess.run(cmd)


@app.command()
def plot_history(
    run_dir: str = typer.Option(
        "runs/current", help="Run directory containing training_history.json"
    ),
    out: str = typer.Option("results/figs/loss_history.png", help="Output path"),
):
    """Plot PINN training loss curves."""
    from src.visualize.training_curve import plot_training_curve

    plot_training_curve(run_dir=run_dir, out_path=out)


@app.command()
def plot_convergence(
    run_dir: str = typer.Option("runs/current", help="Run directory"),
    out: str = typer.Option("results/figs/convergence.png", help="Output path"),
):
    """Plot multi-component convergence (Log scale)."""
    from src.visualize.convergence import plot_convergence as plot_conv

    plot_conv(run_dir=run_dir, out_path=out)


@app.command()
def plot_error_hist(
    model_path: str = typer.Option(
        "checkpoints/final_model.pth", help="Path to trained model"
    ),
    out: str = typer.Option("results/figs/error_hist.png", help="Output path"),
):
    """Plot histogram of GPS phase/azimuth residuals."""
    import torch
    import numpy as np
    from pathlib import Path
    from src.visualize.error_histogram import plot_error_histogram as plot_hist
    from src.core.model import SpatialPINN
    from src.data.loaders import GPSDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialPINN(spatial_dim=3).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    model.eval()

    gps_files = list(Path("data/kinematic_data").glob("gps_strain_*.csv"))
    if not gps_files:
        print("No GPS files found.")
        return
    dataset = GPSDataset([str(f) for f in gps_files])

    with torch.no_grad():
        x = dataset.coords.to(device)
        theta_true = dataset.theta.cpu().numpy()
        out_tensor = model(x)
        sxx = out_tensor[:, 3].cpu().numpy()
        syy = out_tensor[:, 4].cpu().numpy()
        sxy = out_tensor[:, 6].cpu().numpy()
        theta_pred = 0.5 * np.arctan2(2 * sxy, sxx - syy) + np.pi / 2

    diff = theta_pred - theta_true
    errors = np.arctan2(np.sin(diff), np.cos(diff))

    plot_hist(errors=errors, out_path=out, xlabel="Azimuth Error (rad)")


@app.command()
def plot_scatter(
    model_path: str = typer.Option(
        "checkpoints/final_model.pth", help="Path to trained model"
    ),
    out: str = typer.Option("results/figs/scatter.png", help="Output path"),
):
    """Plot predicted vs true azimuth scatter."""
    import torch
    import numpy as np
    from pathlib import Path
    from src.visualize.prediction_scatter import plot_prediction_scatter as plot_scat
    from src.core.model import SpatialPINN
    from src.data.loaders import GPSDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialPINN(spatial_dim=3).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    model.eval()

    gps_files = list(Path("data/kinematic_data").glob("gps_strain_*.csv"))
    if not gps_files:
        print("No GPS files found.")
        return
    dataset = GPSDataset([str(f) for f in gps_files])

    with torch.no_grad():
        x = dataset.coords.to(device)
        theta_true = dataset.theta.cpu().numpy()
        out_tensor = model(x)
        sxx = out_tensor[:, 3].cpu().numpy()
        syy = out_tensor[:, 4].cpu().numpy()
        sxy = out_tensor[:, 6].cpu().numpy()
        theta_pred = 0.5 * np.arctan2(2 * sxy, sxx - syy) + np.pi / 2

    plot_scat(
        y_true=theta_true,
        y_pred=theta_pred,
        out_path=out,
        xlabel="True Azimuth",
        ylabel="Predicted Azimuth",
    )


@app.command()
def plot_cff(
    model_path: str = typer.Option(
        "checkpoints/final_model.pth", help="Path to trained model"
    ),
    depth: float = typer.Option(15.0, help="Depth slice in km"),
    out: str = typer.Option("results/figs/cff_map.png", help="Output path"),
):
    """Plot Coulomb Failure Function map."""
    import torch
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from src.visualize.cff_map import plot_cff_map
    from src.core.model import SpatialPINN
    from src.core.physics import Physics
    from src.data.velocity import VelocityModel
    from src.data.loaders import Transformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialPINN(spatial_dim=3, coupling_enabled=True).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    model.eval()

    gps_files = list(Path("data/kinematic_data").glob("gps_strain_*.csv"))
    if not gps_files:
        print("No GPS files found. Needed for transformer limits.")
        return

    dfs = [pd.read_csv(f) for f in gps_files]
    df = pd.concat(dfs, ignore_index=True)
    bounds = {
        "lon_min": df["lon"].min(),
        "lon_max": df["lon"].max(),
        "lat_min": df["lat"].min(),
        "lat_max": df["lat"].max(),
    }
    vel_model = VelocityModel("data/Morteza_2023/Vel/Pwave.3D.txt", Transformer(bounds))

    res = 100
    lon = np.linspace(bounds["lon_min"], bounds["lon_max"], res)
    lat = np.linspace(bounds["lat_min"], bounds["lat_max"], res)
    Lon, Lat = np.meshgrid(lon, lat)

    x_norm = (Lon.flatten() - bounds["lon_min"]) / (
        bounds["lon_max"] - bounds["lon_min"]
    )
    y_norm = (Lat.flatten() - bounds["lat_min"]) / (
        bounds["lat_max"] - bounds["lat_min"]
    )
    z_norm = (depth - vel_model.min_dep) / (
        vel_model.max_dep - vel_model.min_dep
    ) * 2.0 - 1.0

    pts = np.stack([x_norm, y_norm, np.full_like(x_norm, z_norm)], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device).requires_grad_(True)

    with torch.no_grad():
        cff_val, _ = Physics.coulomb_failure(
            model, pts_t, lambda_p=model.pore_pressure_ratio
        )
        cff_np = cff_val.cpu().numpy().reshape(res, res)

    plot_cff_map(
        x=Lon,
        y=Lat,
        cff=cff_np,
        title=f"Coulomb Failure Function (depth={depth}km)",
        out_path=out,
    )


@app.command()
def results_suite(
    model_path: str = typer.Option("checkpoints/final_model.pth", help="Model path"),
    run_dir: str = typer.Option("runs/current", help="Run directory (for history)"),
):
    """Generate all academic figures."""
    import os

    os.makedirs("results/figs", exist_ok=True)

    print("Generating Training History...")
    plot_history(run_dir=run_dir, out="results/figs/loss_history.png")

    print("Generating Convergence Curves...")
    plot_convergence(run_dir=run_dir, out="results/figs/convergence.png")

    print("Generating Error Histogram...")
    plot_error_hist(model_path=model_path, out="results/figs/error_hist.png")

    print("Generating Prediction Scatter...")
    plot_scatter(model_path=model_path, out="results/figs/scatter.png")

    print("Generating Coulomb Failure Function Map (15km)...")
    plot_cff(model_path=model_path, depth=15.0, out="results/figs/cff_map.png")

    print("Visualization Suite Complete. Check results/figs/ directory.")


@app.command()
def generate(
    regime: str = typer.Option(
        "simple_shear",
        help="Analytical regime (lithostatic, simple_shear, uniaxial_compression)",
    ),
    num_gps: int = typer.Option(500, help="Number of random GPS stations"),
    num_events: int = typer.Option(2000, help="Number of earthquake catalog events"),
    noise_std: float = typer.Option(2.0, help="Gaussian noise std for GPS (degrees)"),
    out_dir: str = typer.Option("data/synthetic", help="Output directory"),
):
    """
    Generate synthetic data (GPS and Catalog) from an analytical tectonic regime.
    """
    from src.validation.synthetic_generator import SyntheticDataGenerator

    print(f"Generating Synthetic Data: Regime={regime}")
    generator = SyntheticDataGenerator()

    # Generate GPS
    out_gps = f"{out_dir}/gps_{regime}.csv"
    generator.generate_gps_data(
        regime=regime,
        num_stations=num_gps,
        noise_std_deg=noise_std,
        out_path=out_gps,
    )
    print(f"✅ GPS data saved to {out_gps} ({num_gps} stations)")

    # Generate Catalog
    out_cat = f"{out_dir}/catalog_{regime}.txt"
    generator.generate_catalog(
        regime=regime,
        num_events=num_events,
        out_path=out_cat,
    )
    print(f"✅ Catalog data saved to {out_cat} ({num_events} events)")

    # Generate Ground Truth Tensor Grid
    out_truth = f"{out_dir}/truth_{regime}.pt"
    generator.generate_ground_truth_grid(
        regime=regime,
        resolution=20,
        out_path=out_truth,
    )
    print(f"✅ Ground truth stress grid saved to {out_truth}")


@app.command("eval-synthetic")
def eval_synthetic_cmd(
    regime: str = typer.Option("simple_shear", help="Analytical tectonic regime"),
    epochs: int = typer.Option(200, help="Training epochs"),
    num_gps: int = typer.Option(500, help="Number of GPS stations"),
    noise_std: float = typer.Option(2.0, help="GPS noise standard deviation"),
    w_seis: float = typer.Option(1.0, help="Seismicity coupling weight"),
):
    """
    Train PINN on synthetic data and compute exact L2 relative stress error.
    """
    from src.analysis.synthetic_eval import evaluate_synthetic_recovery

    evaluate_synthetic_recovery(
        regime=regime,
        num_gps=num_gps,
        noise_std=noise_std,
        epochs=epochs,
        w_seis=w_seis,
    )


@app.command("eval-robustness")
def eval_robustness_cmd(
    regime: str = typer.Option("simple_shear", help="Analytical tectonic regime"),
    epochs: int = typer.Option(200, help="Training epochs"),
    out_file: str = typer.Option(
        "results/tables/robustness.json", help="JSON output path"
    ),
):
    """
    Automate GPS sparsity and noise evaluation sweeps.
    """
    from src.analysis.robustness import run_robustness_sweeps

    run_robustness_sweeps(regime=regime, epochs=epochs, out_file=out_file)


@app.command("eval-ablation")
def eval_ablation_cmd(
    regime: str = typer.Option("simple_shear", help="Analytical tectonic regime"),
    epochs: int = typer.Option(200, help="Training epochs"),
    out_file: str = typer.Option(
        "results/tables/ablation_seismicity.json", help="JSON output path"
    ),
):
    """
    Run physical ablation to prove seismicity coupling resolves magnitude ambiguity.
    """
    from src.analysis.ablation import run_physical_ablation

    run_physical_ablation(regime=regime, epochs=epochs, out_file=out_file)


@app.command("eval-focal")
def eval_focal_cmd(
    model_path: str = typer.Option(
        "checkpoints/final_model.pth", help="Trained PINN weights"
    ),
    focal_mech_file: str = typer.Option(
        "data/kinematic_data/stress_SS.csv", help="Independent focal mechanism CSV"
    ),
    out_file: str = typer.Option(
        "results/tables/focal_errors.json", help="JSON output path"
    ),
):
    """
    Compare PINN stress orientation against independent real-world focal mechanisms.
    """
    from src.analysis.real_world_val import evaluate_against_focal_mechanisms

    evaluate_against_focal_mechanisms(
        model_path=model_path, focal_mech_file=focal_mech_file, out_file=out_file
    )


@app.command("preprocess")
def preprocess_cmd(
    input_file: str = typer.Option(
        "data/files/historical_Eq.txt", help="Raw historical catalog"
    ),
    output_file: str = typer.Option(
        "data/cleaned_historical_Eq.csv", help="Cleaned output path"
    ),
):
    """
    Clean the messy, tab-separated raw earthquake catalog into a standard format.
    """
    from src.data.preprocess import preprocess_catalog

    preprocess_catalog(input_file=input_file, output_file=output_file)


@app.command()
def init_config(
    output: str = "configs/default.yaml",
):
    """
    Generate a default config file.
    """
    cfg = TrainConfig()
    save_train_config(cfg, output)
    print(f"Default config saved to: {output}")


def main():
    app()


if __name__ == "__main__":
    main()
