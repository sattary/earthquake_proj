import torch
import torch.optim as optim
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from src.core.model import SpatialPINN
from src.core.physics import Physics
from src.core.constants import S0, V0
from src.data.loaders import GPSDataset
from src.data.velocity import VelocityModel


class PINNTrainer:
    """
    Trainer class for the Physics-Informed Neural Network.
    Encapsulates life-cycle of training, loss computation, and checkpointing.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        lr: float = 1e-3,
        fourier_scale: float = 10.0,
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = SpatialPINN(
            spatial_dim=spatial_dim, fourier_scale=fourier_scale
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"loss": [], "loss_data": [], "loss_pde": [], "loss_const": []}
        self.transformer = None

    def compute_data_loss(self, x_batch_in, theta_batch):
        """
        Computes azimuthal co-axiality loss.
        """
        spatial_dim = self.model.spatial_dim
        out_data = self.model(x_batch_in)
        sxx_d = out_data[:, 3] if spatial_dim == 3 else out_data[:, 2]
        syy_d = out_data[:, 4] if spatial_dim == 3 else out_data[:, 3]
        sxy_d = out_data[:, 6] if spatial_dim == 3 else out_data[:, 4]

        # Principal Stress Azimuth Prediction (Math Angle: CCW from East)
        theta_pred = 0.5 * torch.atan2(2 * sxy_d, sxx_d - syy_d)

        # GPS Azimuth is Clockwise from North.
        # Math Angle = pi/2 - GPS Azimuth
        theta_batch_math = torch.pi / 2 - theta_batch

        # Data Loss: maximize co-axiality (minimize sin^2(2*diff))
        return torch.mean(torch.sin(2 * (theta_pred - theta_batch_math)) ** 2)

    def compute_physics_losses_3d(self, x_coll, x_surf, vel_model):
        """
        Computes 3D physics residuals using the Core Physics engine.
        """
        rho = 2700.0
        mu = 30e9
        eta_val = 1e21
        scale_x = self.transformer.scale
        scale_z = 15000.0

        if vel_model is not None:
            scale_z = (vel_model.max_dep - vel_model.min_dep) * 1000.0 / 2.0
            props = vel_model.get_material_properties(
                x_coll[:, 0], x_coll[:, 1], x_coll[:, 2]
            )
            rho_t, mu_t = props["rho"], props["mu"]
            rho = rho_t.to(self.device).view(-1, 1)
            mu = mu_t.to(self.device).view(-1, 1)
            eta_val = mu * 3.1536e11

        res_x, res_y, res_z = Physics.momentum_balance_3d(
            self.model,
            x_coll,
            rho=rho,
            g=9.81,
            scale_x=scale_x,
            scale_z=scale_z,
            stress_scale=S0,
        )
        loss_pde = torch.mean(res_x**2 + res_y**2 + res_z**2)

        r_xx, r_yy, r_zz, r_xy, r_yz, r_xz, r_vol = Physics.constitutive_3d(
            self.model,
            x_coll,
            eta=eta_val,
            scale_x=scale_x,
            scale_z=scale_z,
            stress_scale=S0,
            vel_scale=V0,
        )
        loss_const = torch.mean(
            r_xx**2 + r_yy**2 + r_zz**2 + r_xy**2 + r_yz**2 + r_xz**2 + r_vol**2
        )

        rxz, ryz, rzz = Physics.traction_free_surface(
            self.model, x_surf, stress_scale=S0
        )
        loss_bc = torch.mean(rxz**2 + ryz**2 + rzz**2)

        return loss_pde, loss_const, loss_bc

    def train(
        self,
        gps_files: List[str],
        epochs: int = 20000,
        n_coll: int = 20000,
        w_data: float = 5.0,
        w_pde: float = 1.0,
        w_const: float = 1.0,
        w_bc: float = 1.0,
        velocity_file: Optional[str] = None,
    ):
        self.dataset = GPSDataset(gps_files)
        if len(self.dataset) == 0:
            print("Error: Dataset is empty.")
            return
        self.transformer = self.dataset.transformer
        spatial_dim = self.model.spatial_dim

        vel_model = None
        if spatial_dim == 3 and velocity_file:
            vel_model = VelocityModel(velocity_file, self.transformer)

        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=len(self.dataset), shuffle=True
        )

        min_x = self.dataset.coords[:, 0].min().item()
        max_x = self.dataset.coords[:, 0].max().item()
        min_y = self.dataset.coords[:, 1].min().item()
        max_y = self.dataset.coords[:, 1].max().item()

        pbar = tqdm(range(epochs), desc="Training PINN")
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0

            for x_batch, theta_batch in dataloader:
                x_batch = x_batch.to(self.device)
                theta_batch = theta_batch.to(self.device)

                x_coll = torch.rand(n_coll, spatial_dim, device=self.device)
                x_coll[:, 0] = x_coll[:, 0] * (max_x - min_x) + min_x
                x_coll[:, 1] = x_coll[:, 1] * (max_y - min_y) + min_y
                if spatial_dim == 3:
                    x_coll[:, 2] = x_coll[:, 2] * 2.0 - 1.0
                    x_surf = torch.rand(n_coll // 4, 3, device=self.device)
                    x_surf[:, 0] = x_surf[:, 0] * (max_x - min_x) + min_x
                    x_surf[:, 1] = x_surf[:, 1] * (max_y - min_y) + min_y
                    x_surf[:, 2] = -1.0
                    z_surf = -1.0 * torch.ones(x_batch.shape[0], 1, device=self.device)
                    x_batch_in = torch.cat([x_batch, z_surf], dim=1)
                else:
                    x_surf = None
                    x_batch_in = x_batch

                loss_data = self.compute_data_loss(x_batch_in, theta_batch)

                if spatial_dim == 3:
                    loss_pde, loss_const, loss_bc = self.compute_physics_losses_3d(
                        x_coll, x_surf, vel_model
                    )
                else:
                    res_x, res_y = Physics.momentum_balance_2d(self.model, x_coll)
                    loss_pde = torch.mean(res_x**2 + res_y**2)
                    res_c_xx, res_c_yy, res_c_xy = Physics.constitutive_2d(
                        self.model, x_coll, eta=1.0
                    )
                    loss_const = torch.mean(res_c_xx**2 + res_c_yy**2 + res_c_xy**2)
                    loss_bc = torch.tensor(0.0, device=self.device)
                if spatial_dim == 3:
                    Z_norm = torch.linspace(-1, 1, n_coll).to(self.device).view(-1, 1)
                    # Sample Lat/Lon from dataset to get UTM range
                    lats = self.dataset.data["latitude"].values
                    lons = self.dataset.data["longitude"].values
                    coords_norm = self.dataset.coords

                loss = (
                    w_data * loss_data
                    + w_pde * loss_pde
                    + w_const * loss_const
                    + w_bc * loss_bc
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss = loss.item()

            self.history["loss"].append(epoch_loss)
            self.history["loss_data"].append(loss_data.item())
            self.history["loss_pde"].append(loss_pde.item())
            self.history["loss_const"].append(loss_const.item())
            if spatial_dim == 3:
                if "loss_bc" not in self.history:
                    self.history["loss_bc"] = []
                self.history["loss_bc"].append(loss_bc.item())

            if epoch % 10 == 0:
                pbar.set_postfix(
                    {"Loss": f"{epoch_loss:.4f}", "Dat": f"{loss_data.item():.4f}"}
                )

            if (epoch + 1) % 1000 == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}.pth")

        self.save_model("final_model.pth")

        # Save history
        import json

        history_path = Path("results/tables/training_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")

    def save_model(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
