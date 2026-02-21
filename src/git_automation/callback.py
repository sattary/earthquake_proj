import os
from typing import Optional, Dict
from datetime import datetime

from .environment import is_cloud_environment, verify_cloud_environment
from .state_tracker import StateTracker
from .zip_packer import ZipPacker
from .git_pusher import GitPusher


class AutoPushCallback:
    """
    Callback integrated into the training loop to atomic push
    artifacts to GitHub at regular intervals during long runs.
    """

    def __init__(
        self,
        run_dir: str,
        push_interval: int = 100,
        pat: Optional[str] = None,
        dry_run: bool = False,
        force: bool = False,
        include_checkpoints: bool = True,
        repo_dir: str = ".",
    ):
        self.run_dir = os.path.abspath(run_dir)
        self.push_interval = push_interval
        self.pat = pat
        self.dry_run = dry_run
        self.force = force
        self.repo_dir = os.path.abspath(repo_dir)

        # Verify environment early
        verify_cloud_environment(force=self.force)

        self.tracker = StateTracker(self.run_dir)
        self.packer = ZipPacker(self.run_dir, include_checkpoints)

        # Unique branch name for this run based on start time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = os.path.basename(self.run_dir)
        branch_name = f"results/{run_name}_{timestamp}"

        self.pusher = GitPusher(
            repo_dir=self.repo_dir,
            branch_name=branch_name,
            pat=self.pat,
            dry_run=self.dry_run,
        )

        self.is_initialized = False

    def initialize(self) -> bool:
        """Setup branch and initial Git state before training starts."""
        print(f"[AutoPush] Initializing callback for {self.run_dir}...")
        self.is_initialized = True

        if self.dry_run:
            print(
                "[AutoPush] Running in DRY RUN mode. No actual git pushes will occur."
            )

        return self.pusher.setup_branch()

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: Dict[str, float]
    ) -> bool:
        """
        Called at the end of each epoch.
        Returns True if a push occurred.
        """
        if not self.is_initialized:
            self.initialize()

        if not self.tracker.should_push(epoch, self.push_interval):
            return False

        print(f"\n[AutoPush] Triggering configured interval push at epoch {epoch}...")

        try:
            # 1. Zip artifacts
            zip_path = self.packer.create_zip(epoch, total_epochs)

            # 2. Push to git
            success = self.pusher.push_artifact(
                zip_path=zip_path,
                epoch=epoch,
                total_epochs=total_epochs,
                metrics=metrics,
                is_final=False,
            )

            # 3. Update state if successful
            if success or self.dry_run:
                # Find a primary metric to track if available
                best_metric = list(metrics.values())[0] if metrics else 0.0

                self.tracker.write_state(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    push_interval=self.push_interval,
                    last_push_epoch=epoch,
                    best_metric=best_metric,
                    zip_path=zip_path,
                )
                print(
                    f"[AutoPush] Successfully backed up run {os.path.basename(self.run_dir)} at epoch {epoch}"
                )
            else:
                print(
                    "[AutoPush Error] Failed to push artifacts. Training will continue."
                )

            return success

        except Exception as e:
            print(f"[AutoPush Critical] Push failed with exception: {e}")
            print("Training will continue without pushing.")
            return False

    def on_train_end(self, final_metrics: Dict[str, float]) -> bool:
        """Called when training completely finishes."""
        print("\n[AutoPush] Training complete. Triggering final artifacts push...")

        state = self.tracker.read_state()
        epoch = state.get("total_epochs", 1) if state else 1

        try:
            zip_path = self.packer.create_zip(epoch, epoch)
            success = self.pusher.push_artifact(
                zip_path=zip_path,
                epoch=epoch,
                total_epochs=epoch,
                metrics=final_metrics,
                is_final=True,
            )

            if success or self.dry_run:
                best_metric = list(final_metrics.values())[0] if final_metrics else 0.0
                self.tracker.write_state(
                    epoch=epoch,
                    total_epochs=epoch,
                    push_interval=self.push_interval,
                    last_push_epoch=epoch,
                    best_metric=best_metric,
                    zip_path=zip_path,
                )

            # Optional: Return to main branch if needed locally
            if not is_cloud_environment():
                self.pusher.cleanup()

            return success

        except Exception as e:
            print(f"[AutoPush Critical] Final push failed with exception: {e}")
            return False

    def get_status(self) -> Dict:
        """Get the current tracking status."""
        state = self.tracker.read_state()
        if not state:
            return {"status": "uninitialized or empty"}

        status = state.copy()
        epochs_until_push = self.push_interval - (
            status["epoch"] - status["last_push_epoch"]
        )
        status["epochs_until_next_push"] = max(0, epochs_until_push)
        return status
