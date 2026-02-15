import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.pinn.model import SpatialPINN
from src.pinn.velocity_model import VelocityModel
from src.pinn.data import KinematicData
import typer

app = typer.Typer()


class Visualizer:
    def __init__(self, model_path: str, spatial_dim: int = 3, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = SpatialPINN(spatial_dim=spatial_dim).to(self.device)
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.spatial_dim = spatial_dim

        # Load Transformer from Data (to get bounds/scaling)
        # Hack: Initialize a dummy KinematicData to get the transformer
        # In a real app, successful training should save the transformer state.
        # Here we assume the same data file is available.
        gps_files = list(Path("data/kinematic_data").glob("gps_strain_*.csv"))
        if not gps_files:
            raise FileNotFoundError(
                "No GPS data found to initialize CoordinateTransformer."
            )

        self.dataset = KinematicData([str(f) for f in gps_files])
        self.transformer = self.dataset.transformer

        # Bounds (Physical)
        self.min_x_m = self.transformer.min_x
        self.max_x_m = self.transformer.max_x
        self.min_y_m = self.transformer.min_y
        self.max_y_m = self.transformer.max_y

        # Bounds (Normalized)
        self.min_x_norm = self.dataset.coords[:, 0].min().item()
        self.max_x_norm = self.dataset.coords[:, 0].max().item()
        self.min_y_norm = self.dataset.coords[:, 1].min().item()
        self.max_y_norm = self.dataset.coords[:, 1].max().item()

    def predict_grid(self, depth_km: float, resolution: int = 100):
        """
        Predict stress and velocity on a horizontal grid at fixed depth.
        """
        # Create Grid
        x = np.linspace(self.min_x_norm, self.max_x_norm, resolution)
        y = np.linspace(self.min_y_norm, self.max_y_norm, resolution)
        X, Y = np.meshgrid(x, y)

        # Z coordinate (Normalized)
        # transform depth_km to z_norm.
        # CAUTION: We need to know the z-bounds used during training.
        # In trainer.py, we assumed scale_z = 15km or confirmed via velocity model.
        # Only VelocityModel knows the true Z bounds.
        # Let's assume standard crust: 0 to 40km?
        # Trainer used: scale_z = (max_dep - min_dep) / 2
        # VelocityModel loaded Pwave.3D.txt.
        # Let's load VelocityModel to be sure, or pass it in.
        # For now, let's assume z is normalized [-1, 1].
        # If trainer used full range of velocity model [0, 60km], then -1=0km, 1=60km.
        # Let's map depth_km to z_norm [-1, 1].
        z_min_km = 0.0
        z_max_km = 60.0  # Approximation based on Pwave.3D
        z_norm = (depth_km - z_min_km) / (z_max_km - z_min_km) * 2.0 - 1.0

        Z = np.full_like(X, z_norm)

        # Flatten
        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        pts_tensor = torch.tensor(pts, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model(pts_tensor)

        # Unpack outputs
        # 3D: vx, vy, vz, sxx, syy, szz, sxy, syz, sxz
        S0 = 1e7  # Pa
        sxx = out[:, 3].cpu().numpy() * S0
        syy = out[:, 4].cpu().numpy() * S0
        szz = out[:, 5].cpu().numpy() * S0
        sxy = out[:, 6].cpu().numpy() * S0
        syz = out[:, 7].cpu().numpy() * S0
        sxz = out[:, 8].cpu().numpy() * S0

        return (
            X,
            Y,
            sxx.reshape(X.shape),
            syy.reshape(X.shape),
            szz.reshape(X.shape),
            sxy.reshape(X.shape),
        )

    def plot_stress_map(self, depth_km: float, output_path: str = "stress_map.png"):
        X, Y, sxx, syy, szz, sxy = self.predict_grid(depth_km)

        # Calculate Max Horizontal Stress Direction (SHmax)
        # Azimuth of S1 (Most compressive).
        # Math angle: 0.5 * atan2(2*tau, sig_x - sig_y)
        # Note: In geology, compression is often positive. Here tension is positive (physics convention).
        # So most compressive is MINIMUM numerical value (most negative).
        # S_Hmax is direction of S1 (Most compressive horizontal stress).
        # Angle theta = 0.5 * atan2(2*sxy, sxx - syy) gives direction of Max Principal Stress S1 (Algebraic Max).
        # If Sxx, Syy are negative (compression), Algebraic Max is Least Compressive (Tension axis).
        # So this gives T-axis.
        # P-axis (Compression) is theta + 90.

        theta_T = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        theta_P = theta_T + np.pi / 2

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # P-axis ticks (Compression direction)
        # Quiver plot
        skip = 5
        ax.quiver(
            X[::skip, ::skip],
            Y[::skip, ::skip],
            np.cos(theta_P[::skip, ::skip]),
            np.sin(theta_P[::skip, ::skip]),
            headaxislength=0,
            headlength=0,
            pivot="middle",
            scale=30,
            color="red",
            label="SHmax (Compression)",
        )

        # Background: Mean Stress (Pressure)
        pressure = -(sxx + syy + szz) / 3.0  # Positive = Compression
        c = ax.contourf(X, Y, pressure, levels=20, cmap="viridis")
        plt.colorbar(c, label="Mean Stress (Pa)")

        ax.set_title(
            f"Predicted Stress Field at Depth {depth_km} km\nRed Lines: SHmax Direction"
        )
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.legend()

        plt.savefig(output_path)
        print(f"Saved stress map to {output_path}")
        plt.close()

    def plot_velocity_magnitude(
        self, depth_km: float, output_path: str = "velocity_map.png"
    ):
        """
        Plot Velocity Magnitude |v| at fixed depth.
        """
        X, Y, sxx, syy, szz, sxy = self.predict_grid(depth_km)

        # We need to get V from model, but predict_grid currently only returns stress.
        # Let's modify predict_grid or just do it here.
        # Re-using logic for simplicity.
        resolution = 100
        x = np.linspace(self.min_x_norm, self.max_x_norm, resolution)
        y = np.linspace(self.min_y_norm, self.max_y_norm, resolution)
        X_grid, Y_grid = np.meshgrid(x, y)
        z_min_km = 0.0
        z_max_km = 60.0
        z_norm = (depth_km - z_min_km) / (z_max_km - z_min_km) * 2.0 - 1.0
        Z = np.full_like(X_grid, z_norm)
        pts = np.stack([X_grid.flatten(), Y_grid.flatten(), Z.flatten()], axis=1)
        pts_tensor = torch.tensor(pts, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model(pts_tensor)

        V0 = 1e-9  # m/s
        vx = out[:, 0].cpu().numpy() * V0
        vy = out[:, 1].cpu().numpy() * V0
        vz = out[:, 2].cpu().numpy() * V0
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        v_mag = v_mag.reshape(X_grid.shape)

        fig, ax = plt.subplots(figsize=(10, 8))
        c = ax.contourf(X_grid, Y_grid, v_mag, levels=20, cmap="plasma")
        plt.colorbar(c, label="Velocity Magnitude (m/s)")
        ax.set_title(f"Predicted Velocity Magnitude at Depth {depth_km} km")
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        plt.savefig(output_path)
        print(f"Saved velocity map to {output_path}")
        plt.close()


@app.command()
def plot(
    model_path: str = "checkpoints/final_model.pth",
    depth: float = 10.0,
    output_stress: str = "stress_map.png",
    output_velocity: str = "velocity_map.png",
):
    try:
        vis = Visualizer(model_path)
        vis.plot_stress_map(depth, output_stress)
        vis.plot_velocity_magnitude(depth, output_velocity)
        print("Visualization complete.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    typer.run(plot)
