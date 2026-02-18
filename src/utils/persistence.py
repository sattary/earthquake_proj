import os
import sys
import subprocess
from pathlib import Path

# Try to import Kaggle secrets (only works on Kaggle)
try:
    from kaggle_secrets import UserSecretsClient

    KAGGLE = True
except ImportError:
    KAGGLE = False


def run_command(cmd, cwd=None):
    """Run a shell command and print output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(e.stderr)
        raise


class DataPersistence:
    def __init__(self, repo_dir="."):
        self.repo_dir = Path(repo_dir).resolve()
        self.is_colab = "google.colab" in sys.modules
        self.is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        print(
            f"[Init] Environment: {'Colab' if self.is_colab else 'Kaggle' if self.is_kaggle else 'Local/Other'}"
        )

    def _setup_git_auth(self):
        """Configures Git with the token from secrets."""
        if not self.is_kaggle:
            print("[Info] Not on Kaggle. Assuming local git is configured.")
            return

        print("[Auth] Configuring Git with Kaggle Secrets...")
        try:
            user_secrets = UserSecretsClient()
            token = user_secrets.get_secret("github_token")

            # 1. Configure User
            run_command(
                "git config --global user.email 'kaggle-bot@example.com'",
                cwd=self.repo_dir,
            )
            run_command("git config --global user.name 'Kaggle Bot'", cwd=self.repo_dir)

            # 2. Get Current Remote
            result = subprocess.run(
                "git remote get-url origin",
                shell=True,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
            )
            original_url = result.stdout.strip()

            # 3. Inject Token into URL
            # Format: https://TOKEN@github.com/USER/REPO.git
            if "https://" in original_url:
                auth_url = original_url.replace("https://", f"https://{token}@")
                run_command(f"git remote set-url origin {auth_url}", cwd=self.repo_dir)
                print("[Auth] Git remote configured with token.")
            else:
                print(
                    f"[Warning] Remote URL is not HTTPS: {original_url}. Cannot inject token."
                )

        except Exception as e:
            print(f"[Auth Error] Failed to configure Git: {e}")
            print("Tip: Ensure 'github_token' secret is set in Kaggle.")

    def _check_lfs(self, file_path):
        """Checks if file > 90MB and configures LFS."""
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 90:
            print(f"[LFS] File {file_path} is {size_mb:.1f}MB. Configuring LFS...")
            try:
                # Install LFS if needed
                run_command("git lfs install", cwd=self.repo_dir)

                # Track the file extension or specific file
                filename = os.path.basename(file_path)
                run_command(f"git lfs track '{filename}'", cwd=self.repo_dir)
                run_command("git add .gitattributes", cwd=self.repo_dir)
                print(f"[LFS] Tracking {filename}")
            except Exception as e:
                print(f"[LFS Error] Is git-lfs installed? {e}")

    def persist(self, file_path, commit_message="Auto-save results"):
        """
        Commits and pushes the file to the current repository.
        """
        if not os.path.exists(file_path):
            print(f"[Error] File not found: {file_path}")
            return

        # Ensure we are in a git repo
        if not (self.repo_dir / ".git").exists():
            print(f"[Error] No .git found in {self.repo_dir}")
            return

        self._setup_git_auth()
        self._check_lfs(file_path)

        try:
            print(f"[Git] Adding {file_path}...")
            run_command(f"git add {file_path}", cwd=self.repo_dir)

            print("[Git] Committing...")
            run_command(f"git commit -m '{commit_message}'", cwd=self.repo_dir)

            print("[Git] Pushing...")
            run_command("git push origin HEAD", cwd=self.repo_dir)
            print("[Success] Pushed to GitHub.")

        except Exception as e:
            print(f"[Push Failed] Error: {e}")


if __name__ == "__main__":
    # Example Usage
    p = DataPersistence()
    if len(sys.argv) > 1:
        p.persist(sys.argv[1])
