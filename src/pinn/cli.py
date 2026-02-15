import typer
from typing import List
from .trainer import PINNTrainer

app = typer.Typer(help="Earthquake PINN CLI managed by Typer")


@app.command()
def train(
    epochs: int = typer.Option(1000, help="Number of training epochs"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    n_coll: int = typer.Option(1000, help="Number of physics collocation points"),
    w_data: float = typer.Option(1.0, help="Weight for Data Loss"),
    w_pde: float = typer.Option(0.1, help="Weight for PDE Loss"),
    gpu: bool = typer.Option(True, help="Use GPU if available"),
):
    """
    Train the PINN model on sparse GPS data.
    """
    trainer = PINNTrainer(spatial_dim=2, lr=lr)

    # Files
    gps_files = [
        "data/kinematic_data/gps_strain_rayisi2016.csv",
        "data/kinematic_data/gps_strain_khorrami1390.csv",
    ]

    # Filter existing
    import os

    valid_files = [f for f in gps_files if os.path.exists(f)]

    if not valid_files:
        typer.echo("No GPS files found! Please check data/kinematic_data/")
        raise typer.Exit(code=1)

    typer.echo(f"Starting training on {len(valid_files)} GPS files...")
    trainer.train(valid_files, epochs=epochs, n_coll=n_coll, w_data=w_data, w_pde=w_pde)
    typer.echo("Training complete.")


if __name__ == "__main__":
    app()
