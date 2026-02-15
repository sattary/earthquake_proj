import torch
import torch.optim as optim
import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from src.pinn.model import SpatialPINN
from src.pinn.data import KinematicData
from src.pinn.physics import Physics


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
        """
        dataset = KinematicData(gps_files)
        if len(dataset) == 0:
            print("Error: Dataset is empty.")
            return

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
                x_coll = torch.rand(n_coll, 2, device=self.device)
                x_coll[:, 0] = x_coll[:, 0] * (max_x - min_x) + min_x
                x_coll[:, 1] = x_coll[:, 1] * (max_y - min_y) + min_y

                # --- Forward Pass (Data) ---
                out_data = self.model(x_batch)
                sxx_d = out_data[:, 2]
                syy_d = out_data[:, 3]
                sxy_d = out_data[:, 4]

                # Principal Stress Azimuth Prediction
                theta_pred = 0.5 * torch.atan2(2 * sxy_d, sxx_d - syy_d)

                # Data Loss: maximize alignment (minimize sin^2 difference)
                loss_data = torch.mean(torch.sin(theta_pred - theta_batch) ** 2)

                # --- Forward Pass (Physics - Momentum) ---
                # Non-Dimensionalized (Normalized Derivatives)
                res_x, res_y = Physics.momentum_balance_2d(self.model, x_coll)
                loss_pde = torch.mean(res_x**2 + res_y**2)

                # --- Forward Pass (Physics - Constitutive) ---
                # Enforce Viscous Relation: sigma ~ 2*eta*epsilon_dot
                # Non-Dimensionalized (eta=1 implies sigma matches strain-rate magnitude in normalized units)
                res_c_xx, res_c_yy, res_c_xy = Physics.constitutive_2d(
                    self.model, x_coll, eta=1.0
                )
                loss_const = torch.mean(res_c_xx**2 + res_c_yy**2 + res_c_xy**2)

                # Total Loss
                loss = w_data * loss_data + w_pde * loss_pde + w_const * loss_const

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss = loss.item()

            # Update history and progress bar
            self.history["loss"].append(epoch_loss)
            self.history["loss_data"].append(loss_data.item())
            self.history["loss_pde"].append(loss_pde.item())
            self.history["loss_const"].append(loss_const.item())

            if epoch % 10 == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{epoch_loss:.4f}",
                        "Dat": f"{loss_data.item():.4f}",
                        "PDE": f"{loss_pde.item():.4f}",
                        "Cst": f"{loss_const.item():.4f}",
                    }
                )

        self.save_model("final_model.pth")

    def save_model(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
