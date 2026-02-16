import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import typer
from typing import List, Optional
import os
from src.pinn.data import KinematicData
from src.pinn.velocity_model import VelocityModel

app = typer.Typer()

# Set professional aesthetics
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "figure.dpi": 300,
    }
)


def plot_station_density(dataset: KinematicData, output_dir: Path):
    """
    Visualize spatial distribution and density of GPS stations.
    """
    coords = dataset.coords
    df = pd.DataFrame(coords, columns=["X", "Y"])

    fig, ax = plt.subplots(figsize=(10, 8))

    # KDE Density (Heatmap)
    sns.kdeplot(
        data=df,
        x="X",
        y="Y",
        fill=True,
        cmap="rocket",
        alpha=0.8,
        ax=ax,
        levels=15,
        thresh=0.05,
    )

    # Individual Stations
    sns.scatterplot(
        data=df,
        x="X",
        y="Y",
        color="white",
        s=20,
        edgecolor="black",
        marker="^",
        label="GPS Stations",
        ax=ax,
    )

    ax.set_title("GPS Station Density (Normalized Domain)")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "gps_density.png")
    plt.close()


def plot_azimuth_rose(dataset: KinematicData, output_dir: Path):
    """
    Generate a Rose Diagram of GPS strain azimuths.
    """
    # math_az is pi/2 - gps_az
    # We want to plot the original GPS azimuths (CW from North) for geologists
    # We'll extract them from the raw files or back-calculate

    # For simplicity, we'll back-calculate from math_az to show the geological trend
    math_az = dataset.azimuths_rad
    gps_az_deg = np.degrees(np.pi / 2 - math_az) % 360

    # Histogram of azimuths
    radians = np.deg2rad(gps_az_deg)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # Clockwise

    # Binning
    bins = np.linspace(0.0, 2 * np.pi, 25)
    counts, _ = np.histogram(radians, bins=bins)
    widths = np.diff(bins)

    colors = plt.cm.viridis(counts / counts.max())
    ax.bar(
        bins[:-1],
        counts,
        width=widths,
        bottom=0.0,
        color=colors,
        edgecolor="k",
        alpha=0.7,
    )

    ax.set_title("Distribution of GPS Strain Azimuths (CW from North)", pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "azimuth_rose.png")
    plt.close()


def plot_velocity_slices(vm: VelocityModel, output_dir: Path):
    """
    Generate horizontal tomographic slices of the P-wave velocity model.
    """
    depths = [0, 15, 30]  # km
    res = 150
    x_norm = np.linspace(-1, 1, res)
    y_norm = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x_norm, y_norm)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, depth in enumerate(depths):
        # Sample Material Properties
        Z = np.full_like(X, depth)  # VM uses km for depth sampling?
        # Actually our VM get_material_properties takes normalized X, Y and Depth in KM
        props = vm.get_material_properties(X.flatten(), Y.flatten(), Z.flatten())
        Vp = props["vp"].cpu().numpy().reshape(X.shape)

        ax = axes[i]
        im = ax.contourf(X, Y, Vp, levels=20, cmap="magma")
        ax.set_title(f"Vp at Depth {depth} km")
        ax.set_xlabel("Normalized X")
        if i == 0:
            ax.set_ylabel("Normalized Y")

        plt.colorbar(im, ax=ax, label="Vp (km/s)")

    plt.tight_layout()
    plt.savefig(output_dir / "velocity_slices.png")
    plt.close()


@app.command()
def audit(
    gps_dir: str = "data/kinematic_data",
    velocity_file: str = "data/Morteza_2023/Vel/Pwave.3D.txt",
    output_dir: str = "results/eda",
):
    """
    Run full EDA audit and save high-res figures.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Kinematic Data
    gps_files = list(Path(gps_dir).glob("gps_strain_*.csv"))
    if not gps_files:
        print(f"No GPS files found in {gps_dir}")
        return

    print(f"Loading {len(gps_files)} GPS files...")
    dataset = KinematicData([str(f) for f in gps_files])

    # 2. Load Velocity Model
    print(f"Loading Velocity Model: {velocity_file}")
    vm = VelocityModel(velocity_file, dataset.transformer)

    # 3. Generate Plots
    print("Generating Station Density Map...")
    plot_station_density(dataset, out_path)

    print("Generating Azimuth Rose Diagram...")
    plot_azimuth_rose(dataset, out_path)

    print("Generating Velocity Tomographic Slices...")
    plot_velocity_slices(vm, out_path)

    print(f"EDA Complete. Results saved to {out_path}")


if __name__ == "__main__":
    app()
