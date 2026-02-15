import torch
import torch.optim as optim
import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from src.pinn.model import SpatialPINN
from src.pinn.data import KinematicData
from src.pinn.physics import Physics
from src.pinn.velocity_model import VelocityModel


class PINNTrainer:
    """
    Trainer class for the Physics-Informed Neural Network.

    Attributes:
        model (SpatialPINN): The neural network model.
        device (torch.device): Computation device (CPU or CUDA).
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        history (dict): Dictionary to track loss history.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        lr: float = 1e-3,
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = SpatialPINN(spatial_dim=spatial_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"loss": [], "loss_data": [], "loss_pde": [], "loss_const": []}

    def train(
        self,
        gps_files: List[str],
        epochs: int = 1000,
        n_coll: int = 1000,
        w_data: float = 1.0,
        w_pde: float = 0.1,
        w_const: float = 0.1,
        w_bc: float = 0.1,
        velocity_file: Optional[str] = None,
    ):
        """
        Main training loop.

        Args:
            gps_files (List[str]): List of paths to GPS CSV files.
            epochs (int): Number of training epochs.
            n_coll (int): Number of collocation points for physics loss.
            w_data (float): Weight for data loss.
            w_pde (float): Weight for Momentum PDE loss.
            w_const (float): Weight for Constitutive Law loss.
            w_bc (float): Weight for Boundary Condition loss.
            velocity_file (str): Path to Pwave.3D.txt (required for 3D).
        """
        dataset = KinematicData(gps_files)
        if len(dataset) == 0:
            print("Error: Dataset is empty.")
            return

        spatial_dim = self.model.spatial_dim

        # Load Velocity Model if 3D
        vel_model = None
        if spatial_dim == 3:
            if velocity_file is None:
                print(
                    "Warning: 3D training requested but no velocity_file provided. Using default constant properties."
                )
            else:
                vel_model = VelocityModel(velocity_file, dataset.transformer)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=True
        )

        # Domain bounds for collocation sampling (Normalized Domain)
        # dataset.coords is [-1, 1] tensor
        min_x = dataset.coords[:, 0].min().item()
        max_x = dataset.coords[:, 0].max().item()
        min_y = dataset.coords[:, 1].min().item()
        max_y = dataset.coords[:, 1].max().item()

        print(
            f"Domain (Norm): X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]"
        )
        # print(f"Physical Scale: {scale:.2f} meters") - Implicitly handled by Non-Dimensionalization

        pbar = tqdm(range(epochs), desc="Training PINN")
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0

            for x_batch, theta_batch in dataloader:
                x_batch = x_batch.to(self.device)
                theta_batch = theta_batch.to(self.device)

                # --- Collocation Sampling ---
                # Sample within the normalized bounding box
                x_coll = torch.rand(n_coll, spatial_dim, device=self.device)
                # x_coll is [0, 1]. Map to [min, max] or [-1, 1] bounds.
                # data.coords is [-1, 1]. So map to [-1, 1].
                # Actually min_x/max_x are from data.coords.

                # Rescale X, Y
                x_coll[:, 0] = x_coll[:, 0] * (max_x - min_x) + min_x
                x_coll[:, 1] = x_coll[:, 1] * (max_y - min_y) + min_y

                # Rescale Z if 3D
                if spatial_dim == 3:
                    # Z range [-1, 1] assuming full depth range coverage
                    x_coll[:, 2] = x_coll[:, 2] * 2.0 - 1.0

                # --- Surface Sampling for BCs (if 3D) ---
                if spatial_dim == 3:
                    x_surf = torch.rand(n_coll // 4, 3, device=self.device)
                    x_surf[:, 0] = x_surf[:, 0] * (max_x - min_x) + min_x
                    x_surf[:, 1] = x_surf[:, 1] * (max_y - min_y) + min_y
                    x_surf[:, 2] = -1.0  # Forced at Surface
                else:
                    x_surf = None

                # --- Forward Pass (Data) ---
                # GPS is at surface (z normalized = -1 approx, or from data)
                # If 3D, input to model matches data dimension (which is 2D in KinematicData...)
                # Problem: KinematicData returns (x, y). Model expects (x, y, z).
                # Solution: Augment data batch with z_surface for 3D model.

                if spatial_dim == 3:
                    # Append z=-1 (Surface) to x_batch
                    z_surf = -1.0 * torch.ones(x_batch.shape[0], 1, device=self.device)
                    x_batch_in = torch.cat([x_batch, z_surf], dim=1)
                else:
                    x_batch_in = x_batch

                out_data = self.model(x_batch_in)
                sxx_d = out_data[:, 3] if spatial_dim == 3 else out_data[:, 2]
                syy_d = out_data[:, 4] if spatial_dim == 3 else out_data[:, 3]
                sxy_d = out_data[:, 6] if spatial_dim == 3 else out_data[:, 4]
                # Indices: 2D: [vx,vy,sxx,syy,sxy] (2,3,4)
                # Indices: 3D: [vx,vy,vz,sxx,syy,szz,sxy,syz,sxz] (3,4,6)

                # Principal Stress Azimuth Prediction (Math Angle: CCW from East)
                theta_pred = 0.5 * torch.atan2(2 * sxy_d, sxx_d - syy_d)

                # GPS Azimuth is Clockwise from North.
                # Math Angle = pi/2 - GPS Azimuth
                theta_batch_math = torch.pi / 2 - theta_batch

                # Data Loss: maximize co-axiality (minimize sin^2(2*diff))
                # This allows alignment with EITHER the P-axis or T-axis, resolving the 90-degree ambiguity.
                loss_data = torch.mean(
                    torch.sin(2 * (theta_pred - theta_batch_math)) ** 2
                )

                if spatial_dim == 2:
                    # 2D Momentum
                    scale_x = dataset.transformer.scale
                    res_x, res_y = Physics.momentum_balance_2d(
                        self.model, x_coll
                    )  # , scale=scale_x) # Update Physics 2D signature later?
                    loss_pde = torch.mean(res_x**2 + res_y**2)

                    # 2D Constitutive
                    res_c_xx, res_c_yy, res_c_xy = Physics.constitutive_2d(
                        self.model,
                        x_coll,
                        eta=1.0,  # , scale=scale
                    )
                    loss_const = torch.mean(res_c_xx**2 + res_c_yy**2 + res_c_xy**2)

                elif spatial_dim == 3:
                    # 3D Momentum
                    rho = 2700.0  # Default
                    mu = 30e9  # Default
                    eta_val = 1e21  # Default
                    scale_x = dataset.transformer.scale
                    scale_z = 15000.0  # Default 15km half-width if no vel_model

                    if vel_model is not None:
                        # Calculate vertical scale from velocity model depth range
                        # scale_z = (max_dep_km - min_dep_km) * 1000 / 2
                        scale_z = (vel_model.max_dep - vel_model.min_dep) * 1000.0 / 2.0

                        # Query material properties at x_coll
                        # x_coll is (N, 3) normalized
                        rho_t, mu_t = vel_model.get_material_properties(
                            x_coll[:, 0], x_coll[:, 1], x_coll[:, 2]
                        )
                        rho = rho_t.to(self.device).view(-1, 1)
                        mu = mu_t.to(self.device).view(-1, 1)
                        # Estimate eta from mu with Maxwell time ~10,000 years (3.15e11 s)
                        eta_val = mu * 3.1536e11

                    # Call Physics
                    res_x, res_y, res_z = Physics.momentum_balance_3d(
                        self.model,
                        x_coll,
                        rho=rho,
                        g=9.81,
                        scale_x=scale_x,
                        scale_z=scale_z,
                        S0=1e7,
                    )
                    loss_pde = torch.mean(res_x**2 + res_y**2 + res_z**2)

                    # 3D Constitutive
                    r_xx, r_yy, r_zz, r_xy, r_yz, r_xz, r_vol = Physics.constitutive_3d(
                        self.model,
                        x_coll,
                        eta=eta_val,
                        scale_x=scale_x,
                        scale_z=scale_z,
                        S0=1e7,
                        V0=1e-9,
                    )
                    loss_const = torch.mean(
                        r_xx**2
                        + r_yy**2
                        + r_zz**2
                        + r_xy**2
                        + r_yz**2
                        + r_xz**2
                        + r_vol**2
                    )

                    # 3D Surface BC
                    rxz, ryz, rzz = Physics.traction_free_surface(
                        self.model, x_surf, S0=1e7
                    )
                    loss_bc = torch.mean(rxz**2 + ryz**2 + rzz**2)
                else:
                    loss_bc = 0.0

                # Total Loss
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

            # Update history and progress bar
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
                    {
                        "Loss": f"{epoch_loss:.4f}",
                        "Dat": f"{loss_data.item():.4f}",
                        "PDE": f"{loss_pde.item():.4f}",
                        "Cst": f"{loss_const.item():.4f}",
                    }
                )

            # Save Checkpoints
            if (epoch + 1) % 1000 == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}.pth")

        self.save_model("final_model.pth")

    def save_model(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
