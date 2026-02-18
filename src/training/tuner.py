import optuna
import typer
from typing import List, Optional
import os
import torch
import glob
import json

from src.training.engine import PINNTrainer
from src.core.physics import Physics
from src.data.loaders import GPSDataset
from src.data.velocity import VelocityModel

app = typer.Typer()


class OptunaTrainer(PINNTrainer):
    """
    Subclass of PINNTrainer to enable Optuna integration.
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
        # Suggest Hyperparameters (Restoration: Prioritize Physics & Low-Freq)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        w_pde = trial.suggest_float("w_pde", 0.1, 10.0, log=True)
        w_const = trial.suggest_float("w_const", 0.1, 10.0, log=True)
        w_bc = trial.suggest_float("w_bc", 0.1, 10.0, log=True)
        w_data = 5.0  # Increased fixed data weight for stability

        # Suggested Params
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        dataset = GPSDataset(gps_files)
        if len(dataset) == 0:
            print("Error: Dataset is empty.")
            return
        self.transformer = dataset.transformer
        spatial_dim = self.raw_model.spatial_dim

        vel_model = None
        if spatial_dim == 3 and velocity_file:
            vel_model = VelocityModel(velocity_file, self.transformer)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=True
        )

        min_x = dataset.coords[:, 0].min().item()
        max_x = dataset.coords[:, 0].max().item()
        min_y = dataset.coords[:, 1].min().item()
        max_y = dataset.coords[:, 1].max().item()

        for epoch in range(epochs):
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

                # Use refactored logic from parent class
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

            trial.report(epoch_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return epoch_loss


def run_tuning(
    n_trials: int = 20,
    epochs: int = 200,
    spatial_dim: int = 3,
    velocity_file: Optional[str] = None,
    multi_gpu: bool = False,
):
    gps_files = glob.glob("data/kinematic_data/gps_strain_*.csv")
    if not gps_files:
        print("No GPS files found.")
        return

    def objective(trial):
        # Constrain Fourier Scale to prevent Checkerboarding
        f_scale = trial.suggest_float("f_tune", 0.5, 3.0)
        trainer = OptunaTrainer(
            spatial_dim=spatial_dim, fourier_scale=f_scale, multi_gpu=multi_gpu
        )
        return trainer.train_optuna(
            trial, gps_files, epochs=epochs, n_coll=5000, velocity_file=velocity_file
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Tuning Complete ---")
    print(f"Best Trial Score: {study.best_trial.value:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    study.trials_dataframe().to_csv(
        "results/tables/optuna_tuning_results.csv", index=False
    )

    with open("results/tables/best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print("Saved best params to results/tables/best_params.json")


if __name__ == "__main__":
    typer.run(run_tuning)
