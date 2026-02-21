"""
Optuna-based hyperparameter tuning with multi-process parallel trial support.
"""

from __future__ import annotations

import os
import sys
import json
import glob
from pathlib import Path
from typing import Optional, List, Callable, Any
import copy
import multiprocessing as mp
import time

import optuna
import torch
import typer
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.training.engine import PINNTrainer
from src.core.config import save_train_config


def _log(msg: str) -> None:
    """Unbuffered output for Jupyter/multi-process compatibility."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _create_objective(
    gps_files: List[str],
    tune_epochs: int,
    spatial_dim: int,
    velocity_file: Optional[str],
    constitutive: str,
    coupling_enabled: bool,
):
    """Build an Optuna objective closure decoupled from engine."""

    def objective(trial: optuna.Trial) -> float:
        # 1. Suggest Hyperparameters: Prioritize physics, data weights, learning rates, and spatial frequencies
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        w_pde = trial.suggest_float("w_pde", 0.1, 10.0, log=True)
        w_const = trial.suggest_float("w_const", 0.1, 10.0, log=True)
        w_bc = trial.suggest_float("w_bc", 0.1, 10.0, log=True)
        f_scale = trial.suggest_float("f_tune", 0.5, 3.0)

        # 2. Setup Trainer: Enforce single-GPU isolation for parallel stability
        # The worker sets CUDA_VISIBLE_DEVICES, so 'cuda' seamlessly maps to the isolated card.
        trainer = PINNTrainer(
            spatial_dim=spatial_dim,
            lr=lr,
            fourier_scale=f_scale,
            multi_gpu=False,  # DO NOT USE DataParallel for parallel trials
            constitutive=constitutive,
            coupling_enabled=coupling_enabled,
        )

        # 3. Native Execution
        try:
            trainer.train(
                gps_files=gps_files,
                epochs=tune_epochs,
                n_coll=5000,
                w_data=5.0,  # Fixed empirical data weight
                w_pde=w_pde,
                w_const=w_const,
                w_bc=w_bc,
                velocity_file=velocity_file,
                optuna_trial=trial,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            _log(f"  Trial {trial.number}: Failed with error: {e}")
            raise optuna.TrialPruned()

        # 4. Result Hooking
        if not trainer.history["loss"]:
            raise optuna.TrialPruned()

        return trainer.history["loss"][-1]

    return objective


def _run_trial_worker(
    gpu_id: int,
    gps_files: List[str],
    tune_epochs: int,
    spatial_dim: int,
    velocity_file: Optional[str],
    constitutive: str,
    coupling_enabled: bool,
    study_name: str,
    storage: str,
    n_trials: int,
    worker_id: int,
    total_target: int,
):
    """Worker function to run objective trials continuously on a specific isolated GPU."""
    # Isolate this completely fresh Spawned python process to a single specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    objective = _create_objective(
        gps_files=gps_files,
        tune_epochs=tune_epochs,
        spatial_dim=spatial_dim,
        velocity_file=velocity_file,
        constitutive=constitutive,
        coupling_enabled=coupling_enabled,
    )

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    _log(f"[Worker {worker_id} on GPU {gpu_id}] Starting...")

    trial_count = 0
    while trial_count < n_trials:
        try:
            study_summary = optuna.get_all_study_summaries(storage)
            current_trial_count = sum(
                s.n_trials for s in study_summary if s.study_name == study_name
            )

            if current_trial_count >= total_target:
                _log(
                    f"[Worker {worker_id}] Global target ({current_trial_count}/{total_target}) reached."
                )
                break

            study.optimize(objective, n_trials=1, show_progress_bar=False)
            trial_count += 1
            _log(f"[Worker {worker_id}] Completed local trial {trial_count}")
        except Exception as e:
            _log(f"[Worker {worker_id}] Unexpected Search Error: {e}")
            time.sleep(1)

    _log(f"[Worker {worker_id}] Finished processing.")


def run_tuning(
    n_trials: int = 20,
    epochs: int = 200,
    spatial_dim: int = 3,
    velocity_file: Optional[str] = None,
    multi_gpu: bool = False,
    constitutive: str = "viscous",
    coupling_enabled: bool = False,
    base_cfg: Optional[Any] = None,
    auto_push_callback: Optional[Any] = None,
):
    """
    Entry point for hyperparameter sweeps.
    Implements MedianPruner and SQLite persistence for Kaggle resumability.
    """
    gps_files = glob.glob("data/kinematic_data/gps_strain_*.csv")
    if not gps_files:
        print("Error: No GPS files found in data/kinematic_data/")
        return

    study_name = "earthquake_pinn_hpo"
    optuna_dir = Path("results/optuna")
    optuna_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(optuna_dir / f"{study_name}.db")
    storage = f"sqlite:///{db_path}"

    # Persistent Study Generation
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=2),
    )

    remaining = n_trials - len(study.trials)
    if remaining <= 0:
        print(f"Study already has {len(study.trials)} trials. Optimization complete!")
    else:
        print(
            f"Running {remaining} trials ({len(study.trials)} existing, {n_trials} target)."
        )

        # Hardware Detection
        n_workers = 1
        gpu_ids = [0]
        if multi_gpu and torch.cuda.device_count() > 1:
            n_workers = torch.cuda.device_count()
            gpu_ids = list(range(n_workers))

        if n_workers > 1:
            n_workers = min(n_workers, remaining)
            trials_per_worker = remaining // n_workers
            extra_trials = remaining % n_workers

            print(
                f"Parallel HPO: Spawning {n_workers} objective workers across GPUs {gpu_ids}"
            )

            # Crucial for PyTorch CUDA capability in detached forks
            mp.set_start_method("spawn", force=True)

            processes = []
            for i, gpu_id in enumerate(gpu_ids[:n_workers]):
                worker_trials = trials_per_worker + (1 if i < extra_trials else 0)
                p = mp.Process(
                    target=_run_trial_worker,
                    args=(
                        gpu_id,
                        gps_files,
                        epochs,
                        spatial_dim,
                        velocity_file,
                        constitutive,
                        coupling_enabled,
                        study_name,
                        storage,
                        worker_trials,
                        i,
                        n_trials,
                    ),
                )
                p.start()
                processes.append(p)

            # Await completion
            for p in processes:
                p.join()

            # Update core state from workers
            study = optuna.load_study(study_name=study_name, storage=storage)
        else:
            # Standard Fallback
            print("Running HPO sequentially on main process.")
            objective = _create_objective(
                gps_files,
                epochs,
                spatial_dim,
                velocity_file,
                constitutive,
                coupling_enabled,
            )
            study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("WARNING: No trials completed successfully. Check logs.")
        return study

    # Finalize Analytics Output
    print("\n✅ Tuning Complete.")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best Loss: {study.best_value:.4f}")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(
        "results/tables/optuna_tuning_results.csv", index=False
    )

    with open("results/tables/best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    print("Saved best params to best_params.json and study to SQLite DB.")

    if base_cfg is not None:
        best_cfg = copy.deepcopy(base_cfg)
        best_cfg.optim.lr = study.best_params["lr"]
        best_cfg.loss.w_pde = study.best_params["w_pde"]
        best_cfg.loss.w_const = study.best_params["w_const"]
        best_cfg.loss.w_bc = study.best_params["w_bc"]
        best_cfg.model.fourier_scale = study.best_params["f_tune"]

        best_config_path = str(optuna_dir / "best_config.yaml")

        save_train_config(best_cfg, best_config_path)
        print(f"Exported best config to: {best_config_path}")

    if auto_push_callback is not None:
        try:
            print("Triggering auto-push callback...")
            auto_push_callback()
        except Exception as e:
            print(f"Auto-push callback failed: {e}")


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(run_tuning)
    app()
