import torch
import torch.optim as optim
from pathlib import Path
import os
from typing import List, Optional, Dict
from tqdm import tqdm

from src.core.model import SpatialPINN
from src.core.physics import Physics
from src.core.constants import S0, V0, MU_BASELINE
from src.data.loaders import GPSDataset, CatalogDataset
from src.data.velocity import VelocityModel
from src.git_automation import AutoPushCallback
from src.training.multi_gpu import (
    setup_multi_gpu,
    get_model_state_dict,
    load_model_state_dict,
)


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
        auto_push_callback: Optional[AutoPushCallback] = None,
        multi_gpu: bool = True,
        constitutive: str = "viscous",
        coupling_enabled: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_push_callback = auto_push_callback
        self.constitutive = constitutive
        self.coupling_enabled = coupling_enabled
        self.raw_model = SpatialPINN(
            spatial_dim=spatial_dim,
            fourier_scale=fourier_scale,
            coupling_enabled=coupling_enabled,
        ).to(self.device)

        # Multi-GPU Support via new multi_gpu module
        self.multi_gpu = multi_gpu
        self.model = (
            setup_multi_gpu(self.raw_model)
            if multi_gpu
            else self.raw_model.to(self.device)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "loss": [],
            "loss_data": [],
            "loss_pde": [],
            "loss_const": [],
            "loss_seis": [],
        }
        self.transformer = None

    def compute_data_loss(self, x_batch_in, theta_batch):
        """
        Computes azimuthal co-axiality loss.
        """
        # Get spatial_dim from raw_model (DataParallel doesn't expose it)
        spatial_dim = self.raw_model.spatial_dim
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
        Supports both viscous (Stokes) and elastic (Hooke) constitutive laws.
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

        # Branch on constitutive mode
        if self.constitutive == "elastic":
            # Compute Lame's first parameter from mu:
            # Assuming Poisson ratio nu=0.25 -> lambda = mu
            lam = mu if vel_model is not None else MU_BASELINE
            r_xx, r_yy, r_zz, r_xy, r_yz, r_xz, r_vol = Physics.elastic_constitutive_3d(
                self.model,
                x_coll,
                lam=lam,
                mu=mu if vel_model is not None else MU_BASELINE,
                scale_x=scale_x,
                scale_z=scale_z,
                stress_scale=S0,
                disp_scale=V0,
            )
        else:
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

    def compute_seismicity_loss(self, x_catalog, x_coll):
        """
        Compute the Poisson negative log-likelihood seismicity coupling loss.

        Rationale: The earthquake catalog is a point process. The natural
        loss is -sum(log R(x_i)) + integral(R(x)) where the integral
        is approximated by Monte Carlo over collocation points.

        Args:
            x_catalog: (N_eq, 3) tensor of earthquake locations (normalized).
            x_coll: (N_coll, 3) tensor of collocation points for the integral.

        Returns:
            Scalar loss tensor.
        """
        model = self.raw_model

        # Compute CFF and rate at catalog locations
        cff_cat, sigma_n_cat = Physics.coulomb_failure(
            self.model,
            x_catalog,
            lambda_p=model.pore_pressure_ratio,
            stress_scale=S0,
        )
        rate_cat = Physics.seismicity_rate(
            cff_cat,
            sigma_n_cat,
            model.a_param,
            model.r0,
        )

        # Term 1: -sum(log R(x_i)) at observed earthquake locations
        log_rate = torch.log(rate_cat.clamp(min=1e-10))
        term1 = -torch.mean(log_rate)

        # Term 2: integral of R(x) over domain (Monte Carlo)
        cff_coll, sigma_n_coll = Physics.coulomb_failure(
            self.model,
            x_coll,
            lambda_p=model.pore_pressure_ratio,
            stress_scale=S0,
        )
        rate_coll = Physics.seismicity_rate(
            cff_coll,
            sigma_n_coll,
            model.a_param,
            model.r0,
        )
        term2 = torch.mean(rate_coll)

        return term1 + term2

    def load_checkpoint(self, path: str) -> int:
        """Loads model state from path and returns the epoch number.
        Uses strict=False to handle checkpoints missing coupling parameters."""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        load_model_state_dict(self.model, checkpoint)

        try:
            filename = Path(path).stem
            epoch = int(filename.split("_")[-1])
            return epoch
        except Exception:
            print(
                f"Warning: Could not extract epoch from filename {path}. Starting/Continuing with current epoch counter."
            )
            return 0

    def train(
        self,
        gps_files: List[str],
        epochs: int = 20000,
        n_coll: int = 20000,
        w_data: float = 5.0,
        w_pde: float = 1.0,
        w_const: float = 1.0,
        w_bc: float = 1.0,
        w_seis: float = 0.0,
        velocity_file: Optional[str] = None,
        catalog_file: Optional[str] = None,
        min_magnitude: float = 4.0,
        resume_from_checkpoint: Optional[str] = None,
    ):
        self.dataset = GPSDataset(gps_files)
        if len(self.dataset) == 0:
            print("Error: Dataset is empty.")
            return
        self.transformer = self.dataset.transformer
        spatial_dim = self.raw_model.spatial_dim

        vel_model = None
        if spatial_dim == 3 and velocity_file:
            vel_model = VelocityModel(velocity_file, self.transformer)

        # Load earthquake catalog for seismicity coupling
        catalog_coords = None
        if catalog_file and w_seis > 0.0 and spatial_dim == 3:
            catalog_ds = CatalogDataset(
                catalog_file,
                transformer=self.transformer,
                min_magnitude=min_magnitude,
            )
            if len(catalog_ds) > 0:
                # Pre-build tensor of all catalog event coordinates
                all_x, all_y, all_z, all_mag = [], [], [], []
                for i in range(len(catalog_ds)):
                    cx, cy, cz, cm = catalog_ds[i]
                    all_x.append(cx)
                    all_y.append(cy)
                    all_z.append(cz)
                    all_mag.append(cm)
                catalog_coords = torch.stack(
                    [
                        torch.stack(all_x),
                        torch.stack(all_y),
                        # Map depth [0,1] to normalized z [-1, 0]
                        -torch.stack(all_z),
                    ],
                    dim=1,
                ).to(self.device)
                catalog_coords.requires_grad = False
                print(
                    f"Loaded {len(catalog_ds)} catalog events (Mc >= {min_magnitude})"
                )
            else:
                print("Warning: Catalog dataset is empty after Mc filter.")
                w_seis = 0.0

        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=len(self.dataset), shuffle=True
        )

        min_x = self.dataset.coords[:, 0].min().item()
        max_x = self.dataset.coords[:, 0].max().item()
        min_y = self.dataset.coords[:, 1].min().item()
        max_y = self.dataset.coords[:, 1].max().item()

        start_epoch = 0
        if resume_from_checkpoint:
            if os.path.exists(resume_from_checkpoint):
                start_epoch = self.load_checkpoint(resume_from_checkpoint)
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print(
                    f"Checkpoint {resume_from_checkpoint} not found. Starting from 0."
                )

        # Cosine annealing LR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - start_epoch,
            eta_min=1e-6,
        )

        if self.auto_push_callback:
            self.auto_push_callback.initialize()

        pbar = tqdm(range(start_epoch, epochs), desc="Training PINN")
        for epoch in pbar:
            self.model.train()

            for x_batch, theta_batch in dataloader:
                x_batch = x_batch.to(self.device)
                theta_batch = theta_batch.to(self.device)

                # Prepare Data Batch (Add surface depth if 3D)
                if spatial_dim == 3:
                    z_surf = -1.0 * torch.ones(x_batch.shape[0], 1, device=self.device)
                    x_batch_in = torch.cat([x_batch, z_surf], dim=1)
                else:
                    x_batch_in = x_batch

                self.optimizer.zero_grad()

                # 1. Data Loss (Full Batch - small enough)
                loss_data = self.compute_data_loss(x_batch_in, theta_batch)
                (w_data * loss_data).backward()

                current_epoch_loss = w_data * loss_data.item()
                total_pde = 0.0
                total_const = 0.0
                total_bc = 0.0
                total_seis = 0.0

                # 2. Physics Loss (Gradient Accumulation in Chunks)
                batch_size_physics = 4096
                num_batches = (n_coll + batch_size_physics - 1) // batch_size_physics

                for _ in range(num_batches):
                    # Generate mini-batch of collocation points
                    x_coll = torch.rand(
                        batch_size_physics, spatial_dim, device=self.device
                    )
                    x_coll[:, 0] = x_coll[:, 0] * (max_x - min_x) + min_x
                    x_coll[:, 1] = x_coll[:, 1] * (max_y - min_y) + min_y

                    x_surf = None
                    if spatial_dim == 3:
                        x_coll[:, 2] = x_coll[:, 2] * 2.0 - 1.0
                        # Surface points proportional to volume points
                        n_surf = batch_size_physics // 4
                        x_surf = torch.rand(n_surf, 3, device=self.device)
                        x_surf[:, 0] = x_surf[:, 0] * (max_x - min_x) + min_x
                        x_surf[:, 1] = x_surf[:, 1] * (max_y - min_y) + min_y
                        x_surf[:, 2] = -1.0

                    # Compute Physics Losses for this chunk
                    if spatial_dim == 3:
                        l_pde, l_const, l_bc = self.compute_physics_losses_3d(
                            x_coll, x_surf, vel_model
                        )
                    else:
                        res_x, res_y = Physics.momentum_balance_2d(self.model, x_coll)
                        l_pde = torch.mean(res_x**2 + res_y**2)
                        res_c_xx, res_c_yy, res_c_xy = Physics.constitutive_2d(
                            self.model, x_coll, eta=1.0
                        )
                        l_const = torch.mean(res_c_xx**2 + res_c_yy**2 + res_c_xy**2)
                        l_bc = torch.tensor(0.0, device=self.device)

                    loss_chunk = (
                        w_pde * l_pde + w_const * l_const + w_bc * l_bc
                    ) / num_batches

                    # 3. Seismicity coupling loss (only in 3D with catalog)
                    l_seis = torch.tensor(0.0, device=self.device)
                    if w_seis > 0.0 and catalog_coords is not None and spatial_dim == 3:
                        l_seis = self.compute_seismicity_loss(
                            catalog_coords,
                            x_coll,
                        )
                        loss_chunk = loss_chunk + (w_seis * l_seis) / num_batches

                    loss_chunk.backward()

                    # Accumulate for logging
                    total_pde += l_pde.item()
                    total_const += l_const.item()
                    total_bc += l_bc.item()
                    total_seis += l_seis.item()

                # Step Optimizer + Scheduler
                self.optimizer.step()

                # Logging Stats
                avg_pde = total_pde / num_batches
                avg_const = total_const / num_batches
                avg_bc = total_bc / num_batches
                avg_seis = total_seis / num_batches
                current_epoch_loss += (
                    w_pde * avg_pde
                    + w_const * avg_const
                    + w_bc * avg_bc
                    + w_seis * avg_seis
                )

            scheduler.step()

            self.history["loss"].append(current_epoch_loss)
            self.history["loss_data"].append(loss_data.item())
            self.history["loss_pde"].append(avg_pde)
            self.history["loss_const"].append(avg_const)
            self.history["loss_seis"].append(avg_seis)
            if spatial_dim == 3:
                if "loss_bc" not in self.history:
                    self.history["loss_bc"] = []
                self.history["loss_bc"].append(avg_bc)

            if epoch % 10 == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{current_epoch_loss:.4f}",
                        "Dat": f"{loss_data.item():.4f}",
                        "Seis": f"{avg_seis:.4f}" if w_seis > 0 else "off",
                    }
                )

            if self.auto_push_callback:
                metrics = {
                    "loss": current_epoch_loss,
                    "loss_data": loss_data.item(),
                    "loss_pde": avg_pde,
                    "loss_const": avg_const,
                }
                if w_seis > 0:
                    metrics["loss_seis"] = avg_seis
                self.auto_push_callback.on_epoch_end(epoch, epochs, metrics)

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

        if self.auto_push_callback:
            final_metrics = {
                "loss": self.history["loss"][-1] if self.history["loss"] else 0.0,
                "loss_data": self.history["loss_data"][-1]
                if self.history["loss_data"]
                else 0.0,
            }
            self.auto_push_callback.on_train_end(final_metrics)

    def save_model(self, filename: str):
        path = self.checkpoint_dir / filename
        # Always save the unwrapped model state_dict for cross-platform compatibility
        state_to_save = get_model_state_dict(self.model)
        torch.save(state_to_save, path)
        print(f"Model saved to {path}")
