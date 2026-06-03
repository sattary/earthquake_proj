"""
Hierarchical Configuration System for Earthquake PINN Project.

Type-safe, hierarchical configuration using dataclasses with JSON/YAML loading
support and automatic nested merging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Optional, Union, Any, Dict

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


MM_TO_INCH = 1.0 / 25.4


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_dir: str = "data/kinematic_data"
    gps_pattern: str = "gps_strain_*.csv"
    velocity_file: str = "data/Morteza_2023/Vel/Pwave.3D.txt"
    catalog_file: Optional[str] = None
    val_frac: float = 0.15
    workers: int = 4
    min_magnitude: float = 4.0
    augment: bool = False


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    spatial_dim: int = 3
    fourier_scale: float = 1.0
    mapping_size: int = 256
    activation: str = "silu"
    device: str = "auto"
    use_amp: bool = False


@dataclass
class OptimizationConfig:
    """Configuration for optimization and training schedule."""

    epochs: int = 20000
    n_coll: int = 20000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    eta_min: float = 1e-6


@dataclass
class LossConfig:
    """Configuration for loss function weights."""

    w_data: float = 5.0
    w_pde: float = 1.0
    w_const: float = 1.0
    w_bc: float = 1.0
    w_seis: float = 0.0


@dataclass
class PhysicsConfig:
    """Configuration for physics model parameters."""

    coupling_enabled: bool = False
    a_param: float = 1e-3
    mu_friction: float = 0.75
    lambda_lame: float = 30.0


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""

    run_name: str = "default"
    runs_root: str = "runs"
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000
    seed: int = 42

    @property
    def run_dir(self) -> str:
        return str(Path(self.runs_root) / self.run_name)

    @property
    def vis_dir(self) -> str:
        return str(Path(self.runs_root) / self.run_name / "visuals")


@dataclass
class TrainConfig:
    """Top-level configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    config_file: Optional[str] = None
    resume: bool = False
    multi_gpu: bool = True


def _update_dataclass(dc: Any, values: Dict[str, Any]) -> Any:
    """Recursively update dataclass from nested mapping."""
    if not isinstance(values, dict):
        return values

    for key, value in values.items():
        if not hasattr(dc, key):
            continue

        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(dc, key, _update_dataclass(current, value))
        else:
            setattr(dc, key, value)

    return dc


def _load_mapping(path: Path) -> Dict[str, Any]:
    """Load dict from JSON or YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install pyyaml"
            )
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    elif suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)

    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. Use .json or .yaml"
        )


def load_train_config(path: Optional[Union[str, Path]] = None) -> TrainConfig:
    """
    Load config from JSON/YAML file or return defaults if no path provided.

    Args:
        path: Optional path to config file (JSON or YAML)

    Returns:
        TrainConfig with loaded or default values
    """
    cfg = TrainConfig()

    if path is None:
        return cfg

    path = Path(path)
    values = _load_mapping(path)

    return _update_dataclass(cfg, values)


def config_to_yaml(cfg: TrainConfig) -> str:
    """
    Serialize config to YAML string for saving.

    Args:
        cfg: TrainConfig instance

    Returns:
        YAML string representation
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for YAML export. Install with: pip install pyyaml"
        )

    cfg_dict = asdict(cfg)

    cfg_dict.pop("config_file", None)
    cfg_dict.pop("resume", None)
    cfg_dict.pop("multi_gpu", None)

    return yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False)


def save_train_config(cfg: TrainConfig, path: Union[str, Path]) -> None:
    """
    Save config to YAML file.

    Args:
        cfg: TrainConfig instance
        path: Output file path
    """
    path = Path(path)
    yaml_str = config_to_yaml(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_str)
