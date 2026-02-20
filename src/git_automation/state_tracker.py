import json
import os
from datetime import datetime
from typing import Optional, Dict


class StateTracker:
    """
    Tracks training state to disk for resume capability.
    JSON-based persistence of epochs, metrics, and artifact paths.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.state_file = os.path.join(run_dir, ".push_state.json")
        os.makedirs(run_dir, exist_ok=True)

    def read_state(self) -> Optional[Dict]:
        """Read state from disk if it exists."""
        if not os.path.exists(self.state_file):
            return None

        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[Warning] Corrupted state file at {self.state_file}. Ignoring.")
            return None

    def write_state(
        self,
        epoch: int,
        total_epochs: int,
        push_interval: int,
        last_push_epoch: int,
        best_metric: float,
        zip_path: Optional[str] = None,
    ) -> None:
        """Write current state to disk."""
        state = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "push_interval": push_interval,
            "last_push_epoch": last_push_epoch,
            "best_metric": best_metric,
            "last_updated": datetime.now().isoformat(),
            "zip_path": zip_path,
        }

        # Write to temp file then rename for atomic write
        temp_file = self.state_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, self.state_file)

    def should_push(self, current_epoch: int, push_interval: int) -> bool:
        """Check if a push should happen at this epoch."""
        state = self.read_state()
        if not state:
            return current_epoch >= push_interval

        last_push = state.get("last_push_epoch", 0)
        return (current_epoch - last_push) >= push_interval

    def get_resume_epoch(self) -> int:
        """Get the epoch to resume from."""
        state = self.read_state()
        if not state:
            return 0
        return state.get("epoch", 0)

    def is_finished(self) -> bool:
        """Check if training formally finished (reached total_epochs)."""
        state = self.read_state()
        if not state:
            return False

        epoch = state.get("epoch", 0)
        total = state.get("total_epochs", 1)
        return epoch >= total

    def validate_consistency(self, expected_total: int, expected_interval: int) -> bool:
        """
        Check if previous run matches current configurations.
        Prevents resuming with incompatible settings.
        """
        state = self.read_state()
        if not state:
            return True

        # We allow intervals to change, but total epochs shouldn't shrink below current
        recorded_total = state.get("total_epochs", expected_total)
        recorded_epoch = state.get("epoch", 0)

        return recorded_epoch <= expected_total
