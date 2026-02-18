import os
import sys
import subprocess
from pathlib import Path
import importlib
import importlib.util


# --- Kaggle Secrets Loader ---
# kaggle_secrets is a system-level package on Kaggle (e.g. /opt/conda/lib/python3.10/...).
# When running under `uv run`, the venv isolates us from system packages.
# We must search known Kaggle system paths explicitly.
_KAGGLE_SYSTEM_PATHS = [
    "/opt/conda/lib/python3.10/site-packages",
    "/opt/conda/lib/python3.11/site-packages",
    "/opt/conda/lib/python3.12/site-packages",
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/local/lib/python3.11/dist-packages",
    "/usr/local/lib/python3.12/dist-packages",
]


def _load_kaggle_secrets():
    """Load UserSecretsClient, searching system paths if venv import fails."""
    # 1. Try normal import (works if not in a venv, or package is somehow available)
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient
    except ImportError:
        pass

    # 2. Not on Kaggle at all? Skip.
    if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
        return None

    # 3. We ARE on Kaggle but inside a uv venv. Search system paths.
    print("[Import] Searching system paths for kaggle_secrets...")
    for search_dir in _KAGGLE_SYSTEM_PATHS:
        candidate = os.path.join(search_dir, "kaggle_secrets.py")
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("kaggle_secrets", candidate)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    print(f"[Import] Found kaggle_secrets at {candidate}")
                    return mod.UserSecretsClient
                except Exception as exc:
                    print(f"[Import] Failed to load from {candidate}: {exc}")
                    continue

    # 4. Last resort: use subprocess to ask system Python directly
    try:
        result = subprocess.run(
            [
                sys.executable.replace(".venv/bin/python", "/opt/conda/bin/python"),
                "-c",
                "import kaggle_secrets; print(kaggle_secrets.__file__)",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            secrets_path = result.stdout.strip()
            spec = importlib.util.spec_from_file_location(
                "kaggle_secrets", secrets_path
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                print(f"[Import] Found kaggle_secrets via subprocess at {secrets_path}")
                return mod.UserSecretsClient
    except Exception:
        pass

    print("[Import] WARNING: Could not find kaggle_secrets anywhere.")
    return None


UserSecretsClient = _load_kaggle_secrets()


def _get_github_token():
    """Get github_token from Kaggle Secrets, with subprocess fallback."""
    if UserSecretsClient is not None:
        return UserSecretsClient().get_secret("github_token")

    # Fallback: call system Python to get the secret
    try:
        result = subprocess.run(
            [
                "/opt/conda/bin/python",
                "-c",
                "from kaggle_secrets import UserSecretsClient; "
                "print(UserSecretsClient().get_secret('github_token'))",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


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
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error output: {e.stderr}")
        raise


class DataPersistence:
    def __init__(self, repo_dir="."):
        self.repo_dir = Path(repo_dir).resolve()
        self.is_colab = "google.colab" in sys.modules
        self.is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        print(
            f"[Init] Environment: "
            f"{'Colab' if self.is_colab else 'Kaggle' if self.is_kaggle else 'Local/Other'}"
        )

    def _setup_git_auth(self):
        """Configures Git with the GitHub PAT from Kaggle Secrets."""
        if not self.is_kaggle:
            print("[Info] Not on Kaggle. Using local git config.")
            return

        print("[Auth] Configuring Git with Kaggle Secrets...")
        try:
            token = _get_github_token()
            if not token:
                print("[Auth Error] Could not retrieve github_token.")
                return

            # 1. Configure User identity
            run_command(
                "git config user.email 'kaggle-bot@example.com'", cwd=self.repo_dir
            )
            run_command("git config user.name 'Kaggle Bot'", cwd=self.repo_dir)

            # 2. Get Current Remote
            result = subprocess.run(
                "git remote get-url origin",
                shell=True,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
            )
            original_url = result.stdout.strip()

            # 3. Inject PAT into URL
            # For Personal Access Tokens (classic): https://TOKEN@github.com/USER/REPO.git
            if "github.com" in original_url:
                # Strip any existing auth from the URL
                if "@" in original_url:
                    clean_url = original_url.split("@")[-1]
                else:
                    clean_url = original_url.replace("https://", "")
                auth_url = f"https://{token}@{clean_url}"

                run_command(f"git remote set-url origin {auth_url}", cwd=self.repo_dir)
                print("[Auth] Git remote configured with PAT.")
            else:
                print(f"[Warning] Remote URL is not GitHub HTTPS: {original_url}")

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
