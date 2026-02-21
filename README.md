# Physics-Informed Neural Network (PINN) for 3D Crustal Stress Inversion

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Deep learning framework integrating linear elasticity and rate-and-state friction to invert 3D stress tensors from sparse surface GPS strain rates and seismicity catalogs.**

---

## Overview

This repository contains the official implementation of a 3D Physics-Informed Neural Network (PINN) developed for resolving the crustal stress field of the Iranian plateau. The framework solves the fundamental issue of magnitude ambiguity in purely kinematic stress inversions by strongly coupling the predicted stress field to a robust **Rate-and-State friction model**, utilizing historical earthquake focal mechanisms as a proxy for fault criticality.

This methodology relies on:

1. **Mathematical Validation**: Synthetic Andersonian faulting recovery ($L_2$ mathematical proof).
2. **Coupled PDE Constraints**: Enforcing classical equilibrium ($\nabla \cdot \sigma + \rho g = 0$).
3. **Data-Driven Regularization**: Surface GPS strain-rate azimuth matching.
4. **Cloud-Native Infrastructure**: Resumable multi-GPU (Kaggle/Colab) automated pipelines.

---

## Repository Structure

```
earthquake_proj/
├── src/
│   ├── physics/          # Fundamental equations
│   │   ├── equilibrium.py    # Force balance PDEs
│   │   └── friction.py       # Rate-and-State Coulomb friction
│   │
│   ├── validation/       # Mathematical proofs (Phase 1)
│   │   ├── synthetic_generator.py # Generates fake ground-truth Andersonian profiles
│   │   └── synthetic_benchmark.py # Loss tracking against exact L2 error
│   │
│   ├── analysis/         # Nature Paper evaluation suite (Phase 2)
│   │   ├── synthetic_eval.py    # Baseline mathematical validation
│   │   ├── robustness.py        # GPS noise & sparsity sweep grids
│   │   ├── ablation.py          # Seismicity coupling constraint testing
│   │   └── real_world_val.py    # Hold-out validation on Professor's datasets
│   │
│   ├── data/             # Real-world integration (Phase 3)
│   │   ├── loaders.py           # Deep Earth & Surface dataset wrappers
│   │   ├── preprocess.py        # Cleans raw historical_Eq.txt catalogs
│   │   └── transformers.py      # Lat/Lon to localized UTM coordinates
│   │
│   ├── training/         # Core learning loops
│   │   ├── engine.py            # Primary PINNTrainer with PDE backprop
│   │   └── multi_gpu.py         # Kaggle 2x T4 DataParallel wrappers
│   │
│   └── git_automation/   # MLOps Cloud Persistence (Phase 4)
│
├── configs/              # Hierarchical configurations
│   ├── default.yaml         # Synthetic defaults
│   └── real_world.yaml      # Iranian dataset bounding boxes and weights
│
├── notebooks/            # Executable deployment sequences
│   └── run_on_cloud.ipynb   # The definitive Kaggle execution point
│
└── data/                 # Raw Datasets
    ├── kinematic_data/      # GPS strain observations
    ├── files/               # Historical catalogs
    └── Morteza_2023/Vel/    # 3D Velocity Models (Vp)
```

---

## Installation

This repository employs strict dependency parsing using `uv`.

```bash
# Clone the repository
git clone https://github.com/sattary/earthquake_proj.git
cd earthquake_proj

# Install the uv package manager and sync environments
pip install uv
uv sync
```

---

## Recommended Workflow

### Step 1: Preprocess Real Data

Clean the raw professor-provided datasets into consistent tensors.

```bash
uv run earthquake-proj preprocess --input-file data/files/historical_Eq.txt
```

### Step 2: Automated Hyperparameter Tuning & Training

Execute multiprocessed Optuna to sweep the coupling weights (`w_seis`, `w_data`, `w_pde`). The execution will automatically drop into long-running 20,000-epoch real-world training once the best model is identified.

```bash
uv run earthquake-proj tune \
    --n-trials 50 \
    --tune-epochs 500 \
    --auto-push \
    --train-after
```

### Step 3: Manual PINN Inversion (Optional Bypass)

If bypassing Optuna tuning, execute the deep multi-gpu training over the full magnitude of the Iranian velocity model.

```bash
uv run earthquake-proj train \
    --config configs/real_world.yaml \
    --epochs 20000 \
    --run-name "iranian_coupled_model" \
    --multi-gpu \
    --device cuda
```

### Step 3: Run the Evaluation Suite

Compute the mathematical results for publication arrays.

```bash
# 1. Ablation on w_seis (Produces tabular comparison)
uv run earthquake-proj eval-ablation --epochs 1000

# 2. Automated GPS noise sweeps
uv run earthquake-proj eval-robustness --epochs 500

# 3. Validation on hold-out focal datasets
uv run earthquake-proj eval-focal --run-name iranian_coupled_model
```

---

## Cloud Training (Kaggle/Colab)

Due to the extreme bounds of the Iranian geometry, training requires long-running persistence. The framework includes an aggressive **Git LFS Auto-Push** mechanic.

1. Navigate to your Kaggle console.
2. Import `notebooks/run_on_cloud.ipynb`.
3. Fill your `GITHUB_PAT` via Kaggle Secrets.
4. Run All.

The network will automatically save its state dictionary, Git commit all produced models, and zip its output to a discrete branch every `X` epochs. If Kaggle timeouts after 12 hours, use `--resume` natively to continue exactly where the optimizer halted.

---

## Technical Details

### Physics Configuration `(configs/real_world.yaml)`

- `w_seis`: Coupling multiplier that drives Rate-And-State backpropagation. Setting this to > 0 validates the necessity of the earthquake catalog proxy.
- `w_pde`: Standard PDE elasticity mapping.
- `fourier_scale`: Regulates spatial frequency harmonics in the embedding layer to overcome PINN spectral bias at high depths.

## License

MIT
