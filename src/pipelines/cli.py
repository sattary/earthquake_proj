import typer
import glob
import os
import json
from typing import Optional

from src.training.engine import PINNTrainer
from src.training.tuner import run_tuning
from src.pipelines.eda import audit as run_audit
from src.pipelines.inference import app as inference_app

app = typer.Typer(help="L3 Earthquake PINN Operational Pipeline")


@app.command()
def train(
    epochs: int = typer.Option(20000, help="Number of training epochs"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    n_coll: int = typer.Option(20000, help="Number of physics collocation points"),
    w_data: float = typer.Option(5.0, help="Weight for Data Loss"),
    w_pde: float = typer.Option(1.0, help="Weight for PDE Loss"),
    w_const: float = typer.Option(1.0, help="Weight for Constitutive Law Loss"),
    w_bc: float = typer.Option(1.0, help="Weight for Boundary Condition Loss"),
    fourier_scale: float = typer.Option(1.0, help="Fourier Feature scale"),
    spatial_dim: int = typer.Option(3, help="Spatial Dimension (2 or 3)"),
    velocity_file: Optional[str] = typer.Option(
        "data/Morteza_2023/Vel/Pwave.3D.txt", help="Path to Velocity Model"
    ),
    config: Optional[str] = typer.Option(None, help="Path to best_params.json"),
):
    """
    Train the PINN model using the modular core engine.
    """
    if config:
        with open(config, "r") as f:
            params = json.load(f)
            print(f"Loading config from {config}")
            lr = params.get("lr", lr)
            w_pde = params.get("w_pde", w_pde)
            w_const = params.get("w_const", w_const)
            w_bc = params.get("w_bc", w_bc)
            fourier_scale = params.get("fourier_scale", fourier_scale)

    trainer = PINNTrainer(spatial_dim=spatial_dim, lr=lr, fourier_scale=fourier_scale)
    gps_files = glob.glob("data/kinematic_data/gps_strain_*.csv")

    if not gps_files:
        print("Error: No GPS files found.")
        raise typer.Exit(code=1)

    trainer.train(
        gps_files,
        epochs=epochs,
        n_coll=n_coll,
        w_data=w_data,
        w_pde=w_pde,
        w_const=w_const,
        w_bc=w_bc,
        velocity_file=velocity_file,
    )


@app.command()
def tune(
    trials: int = typer.Option(20, help="Number of Optuna trials"),
    epochs: int = typer.Option(200, help="Epochs per trial"),
    spatial_dim: int = typer.Option(3, help="Spatial Dimension"),
    velocity_file: Optional[str] = typer.Option(
        "data/Morteza_2023/Vel/Pwave.3D.txt", help="Path to Velocity Model"
    ),
):
    """
    Run Hyperparameter Tuning.
    """
    run_tuning(
        n_trials=trials,
        epochs=epochs,
        spatial_dim=spatial_dim,
        velocity_file=velocity_file,
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
    fourier_scale: float = 10.0,
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
    fourier_scale: float = typer.Option(1.0, help="Fourier Feature scale"),
):
    """
    Generate the complete academic figure package (Panels, Profiles, Loss).
    """
    from src.pipelines.inference import results_suite as run_suite

    run_suite(model_path=model_path, fourier_scale=fourier_scale)


if __name__ == "__main__":
    app()
