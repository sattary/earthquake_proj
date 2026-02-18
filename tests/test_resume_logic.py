import unittest
import torch
import shutil
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        """Test if the on_checkpoint_save callback is fired."""
        mock_callback = MagicMock()
        trainer = PINNTrainer(
            checkpoint_dir=str(self.checkpoint_dir), on_checkpoint_save=mock_callback
        )

        # Mock everything to speed up
        trainer.model = MagicMock()
        trainer.multi_gpu = False

        # Trigger save
        with patch("src.training.engine.torch.save") as mock_save:
            trainer.save_model("checkpoint_epoch_50.pth")

        expected_path = self.checkpoint_dir / "checkpoint_epoch_50.pth"

        # Verify callback was called with correct path
        mock_callback.assert_called_once()
        args, _ = mock_callback.call_args
        self.assertTrue(str(expected_path) in str(args[0]))


if __name__ == "__main__":
    unittest.main()
