import optuna
import typer
from typing import List, Optional
import os
from src.pinn.trainer import PINNTrainer
import torch
import glob

app = typer.Typer()


class OptunaTrainer(PINNTrainer):
    """
    Subclass of PINNTrainer to enable Optuna integration.
    Overwrites train method to report intermediate values and support pruning.
    """

    def train_optuna(
        self,
        trial: optuna.trial.Trial,
        gps_files: List[str],
        epochs: int = 500,
        n_coll: int = 1000,
        velocity_file: Optional[str] = None,
    ) -> float:
        # Suggest Hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        w_pde = trial.suggest_float("w_pde", 1e-5, 1.0, log=True)
        w_const = trial.suggest_float("w_const", 1e-5, 1.0, log=True)
        w_bc = trial.suggest_float("w_bc", 1e-3, 100.0, log=True)
        # w_data is fixed at 1.0 as a reference anchor
        w_data = 1.0

        # Re-initialize optimizer with suggested LR
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Basic Setup (Data Loading)
        # For tuning, we use a smaller subset or just run for fewer epochs
        from src.pinn.data import KinematicData
        from src.pinn.physics import Physics
        from src.pinn.velocity_model import VelocityModel
        from tqdm import tqdm

        dataset = KinematicData(gps_files)
        # Just use cpu for data processing, model is on self.device

        # Velocity Model
        vel_model = None
        if self.model.spatial_dim == 3 and velocity_file:
            vel_model = VelocityModel(velocity_file, dataset.transformer)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=True
        )

        # Domain Bounds
        min_x = dataset.coords[:, 0].min().item()
        max_x = dataset.coords[:, 0].max().item()
        min_y = dataset.coords[:, 1].min().item()
        max_y = dataset.coords[:, 1].max().item()

        # Training Loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for x_batch, theta_batch in dataloader:
                x_batch = x_batch.to(self.device)
                theta_batch = theta_batch.to(self.device)

                # Collocation
                x_coll = torch.rand(n_coll, self.model.spatial_dim, device=self.device)
                x_coll[:, 0] = x_coll[:, 0] * (max_x - min_x) + min_x
                x_coll[:, 1] = x_coll[:, 1] * (max_y - min_y) + min_y
                if self.model.spatial_dim == 3:
                    x_coll[:, 2] = x_coll[:, 2] * 2.0 - 1.0  # Z norm

                # Surface BC
                x_surf = None
                if self.model.spatial_dim == 3:
                    x_surf = torch.rand(n_coll // 4, 3, device=self.device)
                    x_surf[:, 0] = x_surf[:, 0] * (max_x - min_x) + min_x
                    x_surf[:, 1] = x_surf[:, 1] * (max_y - min_y) + min_y
                    x_surf[:, 2] = -1.0  # Surface

                # Forward Data
                if self.model.spatial_dim == 3:
                    z_surf = -1.0 * torch.ones(x_batch.shape[0], 1, device=self.device)
                    x_batch_in = torch.cat([x_batch, z_surf], dim=1)
                else:
                    x_batch_in = x_batch

                out_data = self.model(x_batch_in)

                # Indices and Loss Calc (duplicated from trainer.py for clarity, could be refactored)
                if self.model.spatial_dim == 3:
                    sxx_d, syy_d, sxy_d = out_data[:, 3], out_data[:, 4], out_data[:, 6]
                else:
                    sxx_d, syy_d, sxy_d = out_data[:, 2], out_data[:, 3], out_data[:, 4]

                theta_pred = 0.5 * torch.atan2(2 * sxy_d, sxx_d - syy_d)
                theta_math = torch.pi / 2 - theta_batch
                loss_data = torch.mean(torch.sin(2 * (theta_pred - theta_math)) ** 2)

                # PDE Loss
                loss_pde = 0.0
                loss_const = 0.0
                loss_bc = 0.0

                scale_x = dataset.transformer.scale

                if self.model.spatial_dim == 3:
                    scale_z = 15000.0
                    rho_val, mu_val = 2700.0, 30e9
                    eta_val = 1e21

                    if vel_model:
                        scale_z = (vel_model.max_dep - vel_model.min_dep) * 1000.0 / 2.0
                        rho_t, mu_t = vel_model.get_material_properties(
                            x_coll[:, 0], x_coll[:, 1], x_coll[:, 2]
                        )
                        eta_val = mu_t.to(self.device).view(-1, 1) * 3.1536e11
                        rho_val = rho_t.to(self.device).view(
                            -1, 1
                        )  # view for broadcast

                    # Momentum
                    rx, ry, rz = Physics.momentum_balance_3d(
                        self.model,
                        x_coll,
                        rho=rho_val,
                        scale_x=scale_x,
                        scale_z=scale_z,
                    )
                    loss_pde = torch.mean(rx**2 + ry**2 + rz**2)

                    # Constitutive
                    c_loss_tuple = Physics.constitutive_3d(
                        self.model,
                        x_coll,
                        eta=eta_val,
                        scale_x=scale_x,
                        scale_z=scale_z,
                    )
                    # Sum of squares of all constitutive residuals (7 terms)
                    loss_const = sum([torch.mean(r**2) for r in c_loss_tuple]) / 7.0

                    # BC
                    bx, by, bz = Physics.traction_free_surface(self.model, x_surf)
                    loss_bc = torch.mean(bx**2 + by**2 + bz**2)

                total_loss = (
                    w_data * loss_data
                    + w_pde * loss_pde
                    + w_const * loss_const
                    + w_bc * loss_bc
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_loss = total_loss.item()

            # Reporting to Optuna
            trial.report(epoch_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return epoch_loss


def run_tuning(
    n_trials: int = 20,
    epochs: int = 200,
    spatial_dim: int = 3,
    velocity_file: Optional[str] = None,
):
    gps_files = glob.glob("data/kinematic_data/gps_strain_*.csv")
    if not gps_files:
        print("No GPS files found.")
        return

    def objective(trial):
        trainer = OptunaTrainer(
            spatial_dim=spatial_dim, lr=1e-3
        )  # LR overridden in train_optuna
        return trainer.train_optuna(
            trial, gps_files, epochs=epochs, n_coll=1000, velocity_file=velocity_file
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Tuning Complete ---")
    print("Best Trial:")
    print(study.best_trial.params)

    # Save best params to a file for the main CLI to pick up?
    # Or just print them for the user to copy-paste into the CLI command?
    # For now, print. In a totally automated pipeline, we'd save to params.json.
    import json

    with open("best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print("Saved best params to best_params.json")


if __name__ == "__main__":
    typer.run(run_tuning)
