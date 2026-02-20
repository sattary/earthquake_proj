"""
Tests for the hierarchical configuration system.
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_train_config(self):
        from src.core.config import (
            TrainConfig,
            DataConfig,
            ModelConfig,
            OptimizationConfig,
            LossConfig,
            PhysicsConfig,
            LoggingConfig,
        )

        cfg = TrainConfig()

        assert cfg.data.data_dir == "data/kinematic_data"
        assert cfg.data.gps_pattern == "gps_strain_*.csv"
        assert cfg.model.spatial_dim == 3
        assert cfg.optim.lr == 1e-3
        assert cfg.loss.w_data == 5.0
        assert cfg.physics.constitutive == "viscous"
        assert cfg.logging.run_name == "default"

    def test_computed_properties(self):
        from src.core.config import LoggingConfig

        logging_cfg = LoggingConfig(run_name="experiment_001", runs_root="runs")

        assert logging_cfg.run_dir == "runs/experiment_001"
        assert logging_cfg.vis_dir == "runs/experiment_001/visuals"


class TestLoadJSONConfig:
    """Test loading from JSON config files."""

    def test_load_partial_json(self, tmp_path):
        from src.core.config import load_train_config, TrainConfig

        config_data = {
            "optim": {"epochs": 500, "lr": 1e-4},
            "logging": {"run_name": "test_run"},
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        cfg = load_train_config(str(config_file))

        assert cfg.optim.epochs == 500
        assert cfg.optim.lr == 1e-4
        assert cfg.optim.n_coll == 20000
        assert cfg.logging.run_name == "test_run"

    def test_load_nested_json(self, tmp_path):
        from src.core.config import load_train_config

        config_data = {
            "model": {"spatial_dim": 2, "fourier_scale": 5.0},
            "physics": {"constitutive": "elastic", "coupling_enabled": True},
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        cfg = load_train_config(str(config_file))

        assert cfg.model.spatial_dim == 2
        assert cfg.model.fourier_scale == 5.0
        assert cfg.physics.constitutive == "elastic"
        assert cfg.physics.coupling_enabled is True


class TestLoadYAMLConfig:
    """Test loading from YAML config files."""

    def test_load_partial_yaml(self, tmp_path):
        from src.core.config import load_train_config

        config_yaml = """
optim:
  epochs: 300
  lr: 0.0005

logging:
  run_name: yaml_test
  seed: 123
"""

        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        cfg = load_train_config(str(config_file))

        assert cfg.optim.epochs == 300
        assert cfg.optim.lr == 5e-4
        assert cfg.logging.run_name == "yaml_test"
        assert cfg.logging.seed == 123

    def test_physics_config_yaml(self, tmp_path):
        from src.core.config import load_train_config

        config_yaml = """
physics:
  constitutive: elastic
  coupling_enabled: true
  mu_friction: 0.6
"""

        config_file = tmp_path / "physics.yaml"
        config_file.write_text(config_yaml)

        cfg = load_train_config(str(config_file))

        assert cfg.physics.constitutive == "elastic"
        assert cfg.physics.coupling_enabled is True
        assert cfg.physics.mu_friction == 0.6


class TestSerialization:
    """Test config serialization."""

    def test_config_to_yaml(self):
        from src.core.config import TrainConfig, config_to_yaml

        cfg = TrainConfig()
        cfg.logging.run_name = "serialization_test"
        cfg.optim.epochs = 999

        yaml_str = config_to_yaml(cfg)

        assert "epochs: 999" in yaml_str
        assert "run_name: serialization_test" in yaml_str

    def test_save_and_load_roundtrip(self, tmp_path):
        from src.core.config import TrainConfig, save_train_config, load_train_config

        cfg = TrainConfig()
        cfg.logging.run_name = "roundtrip_test"
        cfg.optim.epochs = 777
        cfg.physics.coupling_enabled = True

        config_path = tmp_path / "roundtrip.yaml"
        save_train_config(cfg, config_path)

        loaded_cfg = load_train_config(config_path)

        assert loaded_cfg.logging.run_name == "roundtrip_test"
        assert loaded_cfg.optim.epochs == 777
        assert loaded_cfg.physics.coupling_enabled is True


class TestConfigMerge:
    """Test hierarchical config merging."""

    def test_merge_preserves_defaults(self, tmp_path):
        from src.core.config import load_train_config

        config_data = {"data": {"min_magnitude": 5.0}}

        config_file = tmp_path / "merge.json"
        config_file.write_text(json.dumps(config_data))

        cfg = load_train_config(str(config_file))

        assert cfg.data.min_magnitude == 5.0
        assert cfg.data.val_frac == 0.15
        assert cfg.model.spatial_dim == 3

    def test_unknown_fields_ignored(self, tmp_path):
        from src.core.config import load_train_config

        config_data = {
            "unknown_section": {"unknown_field": "value"},
            "optim": {"epochs": 100},
        }

        config_file = tmp_path / "unknown.json"
        config_file.write_text(json.dumps(config_data))

        cfg = load_train_config(str(config_file))

        assert cfg.optim.epochs == 100


class TestNonePath:
    """Test behavior when no config path is provided."""

    def test_none_returns_defaults(self):
        from src.core.config import load_train_config

        cfg = load_train_config(None)

        assert cfg.optim.epochs == 20000
        assert cfg.logging.run_name == "default"
