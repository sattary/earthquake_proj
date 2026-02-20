"""
CLI for Earthquake PINN Project.

Supports both hierarchical config files (YAML/JSON) and flat CLI options.
"""

import typer
import glob
import os
from pathlib import Path
from typing import Optional

from src.training.engine import PINNTrainer
from src.training.tuner import run_tuning
from src.pipelines.eda import audit as run_audit
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
                except:
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

    run_tuning(
        n_trials=trials or 20,
        epochs=epochs or cfg.optim.epochs,
        spatial_dim=cfg.model.spatial_dim,
        velocity_file=cfg.data.velocity_file,
        multi_gpu=cfg.multi_gpu,
        constitutive=cfg.physics.constitutive,
        coupling_enabled=cfg.physics.coupling_enabled,
    )


@app.command()
def audit(
    gps_dir: str = "data/kinematic_data",
    velocity_file: str = "data/Morteza_2023/Vel/Pwave.3D.txt",
    output_dir: str = "results/eda",
):
    """
    Perform Exploratory Data Analysis (EDA).
    """
    run_audit(gps_dir=gps_dir, velocity_file=velocity_file, output_dir=output_dir)


@app.command()
def plot(
    model_path: str = "checkpoints/final_model.pth",
    depth: float = 10.0,
    fourier_scale: float = 1.0,
    velocity_file: Optional[str] = "data/Morteza_2023/Vel/Pwave.3D.txt",
    output_stress: str = "results/figs/stress_map.png",
    output_velocity: str = "results/figs/velocity_map.png",
):
    """
    Generate physical maps from a trained model.
    """
    from src.pipelines.inference import run_plotting

    run_plotting(
        model_path=model_path,
        depth=depth,
        fourier_scale=fourier_scale,
        velocity_file=velocity_file,
        output_stress=output_stress,
        output_velocity=output_velocity,
    )


@app.command()
def results_suite(
    model_path: str = "checkpoints/final_model.pth",
    fourier_scale: float = 1.0,
):
    """
    Generate the complete academic figure package (Panels, Profiles, Loss).
    """
    from src.pipelines.inference import results_suite as run_suite

    run_suite(model_path=model_path, fourier_scale=fourier_scale)


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
