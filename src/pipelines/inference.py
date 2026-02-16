import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import os
import json
import typer

from src.core.model import SpatialPINN
from src.data.velocity import VelocityModel
from src.data.loaders import GPSDataset
from src.core.constants import S0, V0

app = typer.Typer()

# Academic Style
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


def save_academic_fig(fig, base_path: str):
    """Saves a figure in PDF (vector), SVG (vector), and high-res 600dpi PNG."""
    p = Path(base_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Vector formats
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", transparent=False)
    fig.savefig(p.with_suffix(".svg"), bbox_inches="tight", transparent=False)
    # High-res raster
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=600, transparent=False)
    print(f"Exported academic figure set: {p.stem}.{{pdf,svg,png}} to {p.parent}")


class Visualizer:
    def __init__(
        self,
        model_path: str,
        velocity_file: Optional[str] = None,
        spatial_dim: int = 3,
        fourier_scale: float = 10.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = SpatialPINN(
            spatial_dim=spatial_dim, fourier_scale=fourier_scale
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        gps_files = list(Path("data/kinematic_data").glob("gps_strain_*.csv"))
        self.dataset = GPSDataset([str(f) for f in gps_files])
        self.transformer = self.dataset.transformer

        self.min_x_norm = self.dataset.coords[:, 0].min().item()
        self.max_x_norm = self.dataset.coords[:, 0].max().item()
        self.min_y_norm = self.dataset.coords[:, 1].min().item()
        self.max_y_norm = self.dataset.coords[:, 1].max().item()

        self.z_min_km, self.z_max_km = 0.0, 30.0
        if velocity_file and spatial_dim == 3:
            vm = VelocityModel(velocity_file, self.transformer)
            self.z_min_km, self.z_max_km = vm.min_dep, vm.max_dep

    def predict_grid(self, depth_km: float, resolution: int = 100):
        x = np.linspace(self.min_x_norm, self.max_x_norm, resolution)
        y = np.linspace(self.min_y_norm, self.max_y_norm, resolution)
        X, Y = np.meshgrid(x, y)
        z_norm = (depth_km - self.z_min_km) / (
            self.z_max_km - self.z_min_km
        ) * 2.0 - 1.0
        pts = np.stack(
            [X.flatten(), Y.flatten(), np.full_like(X.flatten(), z_norm)], axis=1
        )
        pts_t = torch.tensor(pts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(pts_t)
        return (
            X,
            Y,
            out[:, 3].cpu().numpy() * S0,
            out[:, 4].cpu().numpy() * S0,
            out[:, 5].cpu().numpy() * S0,
            out[:, 6].cpu().numpy() * S0,
        )

    def plot_stress_panel(self, depths: List[float], output_path: str):
        fig, axes = plt.subplots(
            1, len(depths), figsize=(6 * len(depths), 6), sharey=True
        )
        for i, d in enumerate(depths):
            X, Y, sxx, syy, szz, sxy = self.predict_grid(d)
            pressure = -(sxx + syy + szz) / 3.0 / 1e6  # MPa
            ax = axes[i] if len(depths) > 1 else axes
            im = ax.contourf(X, Y, pressure.reshape(X.shape), levels=20, cmap="viridis")
            theta = 0.5 * np.arctan2(2 * sxy, sxx - syy).reshape(X.shape) + np.pi / 2
            skip = 10
            ax.quiver(
                X[::skip, ::skip],
                Y[::skip, ::skip],
                np.cos(theta[::skip, ::skip]),
                np.sin(theta[::skip, ::skip]),
                color="red",
                headaxislength=0,
                headlength=0,
                pivot="middle",
                scale=30,
            )
            ax.set_title(f"Depth: {d} km")
            plt.colorbar(im, ax=ax, label="Mean Stress (MPa)")
        plt.tight_layout()
        save_academic_fig(fig, output_path)
        plt.close()

    def plot_vertical_profile(self, output_path: str):
        depths = np.linspace(self.z_min_km, self.z_max_km, 50)
        pressures = []
        for d in depths:
            _, _, sxx, syy, szz, _ = self.predict_grid(d, resolution=10)
            pressures.append(-np.mean(sxx + syy + szz) / 3.0 / 1e6)

        plt.figure(figsize=(6, 8))
        plt.plot(pressures, depths, "b-", lw=2, label="PINN Predicted")
        theory = [(2700 * 9.81 * d * 1000) / 1e6 for d in depths]
        plt.plot(theory, depths, "r--", label="Lithostatic (Theory)")
        plt.gca().invert_yaxis()
        plt.xlabel("Mean Stress (MPa)")
        plt.ylabel("Depth (km)")
        plt.title("Vertical Stress Profile Convergence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_academic_fig(plt.gcf(), output_path)
        plt.close()


@app.command()
def plot_history(
    history_json: str = "results/tables/training_history.json",
    output_path: str = "results/figs/loss_history.png",
):
    with open(history_json, "r") as f:
        h = json.load(f)
    plt.figure(figsize=(10, 6))
    for k, v in h.items():
        if len(v) > 0:
            plt.plot(v, label=k.replace("loss_", "").upper())
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("PINN Multi-Component Convergence History")
    plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_academic_fig(plt.gcf(), output_path)
    plt.close()


@app.command()
def results_suite(model_path: str = "checkpoints/final_model.pth"):
    os.makedirs("results/figs", exist_ok=True)
    vis = Visualizer(model_path)
    print("Generating Academic Stress Panel...")
    vis.plot_stress_panel([5.0, 15.0, 25.0], "results/figs/stress_panel_3d.png")
    print("Generating Vertical Stress Profile...")
    vis.plot_vertical_profile("results/figs/vertical_profile.png")
    if os.path.exists("results/tables/training_history.json"):
        print("Generating Loss History...")
        plot_history()
    print("Full Visualization Suite Complete.")


if __name__ == "__main__":
    app()
