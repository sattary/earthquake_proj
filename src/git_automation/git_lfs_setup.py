import subprocess
from .environment import get_environment_name


def setup_git_lfs(repo_dir: str = ".") -> bool:
    """
    Initialize Git LFS and track large files.
    Usually only needs to be run once per project.

    Args:
        repo_dir: Path to the repository

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"[{get_environment_name().upper()}] Setting up Git LFS...")
    patterns = [
        "artifacts/*.zip",
        "*.pth",
        "*.onnx",
        "data/**/*.h5",
        "checkpoints/*.pth",
    ]

    try:
        # Install LFS
        subprocess.run(
            "git lfs install", shell=True, cwd=repo_dir, check=True, capture_output=True
        )

        # Track patterns
        for pattern in patterns:
            subprocess.run(
                f"git lfs track '{pattern}'",
                shell=True,
                cwd=repo_dir,
                check=True,
                capture_output=True,
            )
            print(f"Tracking {pattern} with LFS")

        # Add .gitattributes
        subprocess.run(
            "git add .gitattributes",
            shell=True,
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )

        # Commit LFS setup
        subprocess.run(
            "git commit -m 'chore: configure Git LFS tracking' || echo 'No changes to commit'",
            shell=True,
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )

        print(f"[{get_environment_name().upper()}] Git LFS setup complete.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[Error] Git LFS setup failed: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode('utf-8')}")
        return False


if __name__ == "__main__":
    setup_git_lfs()
