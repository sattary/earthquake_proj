import typer
from typing import List, Optional
from .trainer import PINNTrainer

app = typer.Typer(help="Earthquake PINN CLI managed by Typer")


@app.callback()
def callback():
    """
    Earthquake PINN CLI.
    """


@app.command()
def train(
    epochs: int = typer.Option(1000, help="Number of training epochs"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    n_coll: int = typer.Option(1000, help="Number of physics collocation points"),
    w_data: float = typer.Option(1.0, help="Weight for Data Loss"),
    w_pde: float = typer.Option(1e-4, help="Weight for PDE Loss"),
    w_const: float = typer.Option(1e-4, help="Weight for Constitutive Law Loss"),
    w_bc: float = typer.Option(1e-4, help="Weight for Boundary Condition Loss"),
    gpu: bool = typer.Option(True, help="Use GPU if available"),
    spatial_dim: int = typer.Option(2, help="Spatial Dimension (2 or 3)"),
    velocity_file: Optional[str] = typer.Option(
        None, help="Path to Pwave.3D.txt (required for 3D)"
    ),
):
    """
    Train the PINN model on sparse GPS data.
    """
    trainer = PINNTrainer(spatial_dim=spatial_dim, lr=lr)

    # Files
    # Files - Dynamic Glob
    import glob

    gps_files = glob.glob("data/kinematic_data/gps_strain_*.csv")

    # Filter existing
    import os

    valid_files = [f for f in gps_files if os.path.exists(f)]

    if not valid_files:
        typer.echo("No GPS files found! Please check data/kinematic_data/")
        raise typer.Exit(code=1)

    typer.echo(
        f"Starting training on {len(valid_files)} GPS files (Dim={spatial_dim})..."
    )
    trainer.train(
        valid_files,
        epochs=epochs,
        n_coll=n_coll,
        w_data=w_data,
        w_pde=w_pde,
        w_const=w_const,
        w_bc=w_bc,
        velocity_file=velocity_file,
    )
    typer.echo("Training complete.")


if __name__ == "__main__":
    app()
