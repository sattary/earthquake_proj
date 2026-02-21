import os
import subprocess
from typing import Optional, Dict
from .environment import get_environment_name


class GitPusher:
    """
    Handles git operations (branching, committing, pushing) safely.
    """

    def __init__(
        self,
        repo_dir: str,
        branch_name: str,
        pat: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.repo_dir = repo_dir
        self.branch_name = branch_name
        self.pat = pat
        self.dry_run = dry_run
        self.original_branch = self._get_current_branch()
        self.remote_url = self._get_remote_url()
        self.env_name = get_environment_name().upper()

    def _run_cmd(self, cmd: str) -> tuple[bool, str]:
        """Run a git command and return success status and output."""
        if self.dry_run:
            print(
                f"[DRY-RUN] Would execute: {cmd.replace(self.pat, '***') if self.pat else cmd}"
            )
            return True, ""

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            # Mask PAT in error output
            err_msg = e.stderr.replace(self.pat, "***") if self.pat else e.stderr
            print(f"[Git Error] Command failed: {cmd.split()[0:2]}...")
            print(f"Details: {err_msg}")
            return False, err_msg

    def _get_current_branch(self) -> str:
        """Get the name of the current git branch."""
        try:
            result = subprocess.run(
                "git rev-parse --abbrev-ref HEAD",
                shell=True,
                cwd=self.repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "main"

    def _get_remote_url(self) -> str:
        """Get the origin remote URL."""
        try:
            result = subprocess.run(
                "git remote get-url origin",
                shell=True,
                cwd=self.repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def _setup_auth(self) -> bool:
        """Configure git authentication using PAT."""
        if not self.pat or self.dry_run:
            return True

        print(f"[{self.env_name}] Configuring Git authentication...")

        # Configure user if not set
        self._run_cmd("git config user.email 'kaggle-bot@example.com'")
        self._run_cmd("git config user.name 'Cloud Bot'")

        url = self.remote_url
        if "github.com" in url:
            # Strip existing auth
            if "@" in url:
                clean_url = url.split("@")[-1]
            else:
                clean_url = url.replace("https://", "").replace("http://", "")

            auth_url = f"https://{self.pat}@{clean_url}"
            success, _ = self._run_cmd(f"git remote set-url origin {auth_url}")
            return success
        return False

    def setup_branch(self) -> bool:
        """Create and checkout the results branch for this run."""
        print(f"[{self.env_name}] Setting up branch: {self.branch_name}")

        if self.dry_run:
            print(f"[DRY-RUN] Would setup branch {self.branch_name}")
            return True

        # Check if branch exists
        exists, _ = self._run_cmd(f"git rev-parse --verify {self.branch_name}")

        if exists:
            # Branch exists, just check it out
            success, _ = self._run_cmd(f"git checkout {self.branch_name}")
            return success
        else:
            # Create new branch
            success, _ = self._run_cmd(f"git checkout -b {self.branch_name}")
            return success

    def push_artifact(
        self,
        zip_path: str,
        epoch: int,
        total_epochs: int,
        metrics: Dict,
        is_final: bool = False,
    ) -> bool:
        """Commit and push the zipped artifact."""
        if not self._setup_auth():
            print(
                f"[{self.env_name}] Warning: Could not setup git authentication. Push may fail."
            )

        # Ensure we're on the right branch
        self._run_cmd(f"git checkout {self.branch_name}")

        # Copy zip to a tracked location if it's not already
        tracked_path = zip_path
        if not zip_path.startswith(self.repo_dir):
            # This shouldn't normally happen if zip_packer puts it in artifacts/
            print(f"[Warning] Zip path {zip_path} is outside repo {self.repo_dir}")
            return False

        # Stage the file
        success, _ = self._run_cmd(f"git add {tracked_path}")
        if not success:
            return False

        # Stage the state file so we can resume
        state_file = os.path.join(os.path.dirname(tracked_path), ".push_state.json")
        if os.path.exists(state_file):
            self._run_cmd(f"git add {state_file}")

        # Create commit message
        status = "Final" if is_final else "Progress"
        metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        msg = f"{status}: Epoch {epoch}/{total_epochs} | {metric_str}"

        print(f"[{self.env_name}] Committing: {msg}")
        success, _ = self._run_cmd(f"git commit -m '{msg}'")

        # It's fine if there are no changes to commit
        if not success and "nothing to commit" not in _:
            # Only fail if it's a real error
            pass

        print(f"[{self.env_name}] Pushing to origin/{self.branch_name}...")
        success, err = self._run_cmd(f"git push -u origin {self.branch_name}")

        if success:
            print(f"[{self.env_name}] Push successful!")
        return success

    def cleanup(self) -> None:
        """Return to original branch (useful during local testing)."""
        if self.dry_run:
            print(f"[DRY-RUN] Would checkout {self.original_branch}")
            return

        if self.original_branch:
            print(f"[{self.env_name}] Returning to {self.original_branch}")
            self._run_cmd(f"git checkout {self.original_branch}")

        # Try to restore safe remote URL just in case
        if self.pat and self.remote_url:
            self._run_cmd(f"git remote set-url origin {self.remote_url}")
