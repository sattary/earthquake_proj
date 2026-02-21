import unittest
import torch
import shutil
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.training.engine import PINNTrainer


class TestResumeLogic(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.test_dir) / "checkpoints"
        self.checkpoint_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_epoch_extraction(self):
        """Test if load_checkpoint correctly extracts epoch from filename."""
        trainer = PINNTrainer(checkpoint_dir=str(self.checkpoint_dir))

        # Create a dummy checkpoint
        dummy_path = self.checkpoint_dir / "checkpoint_epoch_123.pth"
        torch.save({"dummy": 1}, dummy_path)

        # Mock load_state_dict to avoid actual model loading issues
        trainer.model = MagicMock()
        trainer.multi_gpu = False

        epoch = trainer.load_checkpoint(str(dummy_path))
        self.assertEqual(epoch, 123)

    def test_epoch_extraction_failure(self):
        """Test fallback when filename doesn't have an epoch."""
        trainer = PINNTrainer(checkpoint_dir=str(self.checkpoint_dir))

        dummy_path = self.checkpoint_dir / "best_model.pth"
        torch.save({"dummy": 1}, dummy_path)

        trainer.model = MagicMock()
        trainer.multi_gpu = False

        epoch = trainer.load_checkpoint(str(dummy_path))
        self.assertEqual(epoch, 0)

    def test_callback_trigger(self):
        """Test if the auto_push_callback is attached correctly."""
        mock_callback = MagicMock()
        trainer = PINNTrainer(
            checkpoint_dir=str(self.checkpoint_dir), auto_push_callback=mock_callback
        )

        # Mock everything to speed up
        trainer.model = MagicMock()
        trainer.multi_gpu = False

        self.assertEqual(trainer.auto_push_callback, mock_callback)


if __name__ == "__main__":
    unittest.main()
