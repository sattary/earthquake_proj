import os
import sys


def is_kaggle() -> bool:
    """Detect if running in Kaggle environment."""
    return (
        os.path.exists("/kaggle")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_CONTAINER_NAME" in os.environ
    )


def is_colab() -> bool:
    """Detect if running in Google Colab environment."""
    return (
        "google.colab" in sys.modules
        or "COLAB_GPU" in os.environ
        or "COLAB_RELEASE_TAG" in os.environ
    )


def is_cloud_environment() -> bool:
    """Detect if running in any supported cloud environment."""
    return is_kaggle() or is_colab()


def get_environment_name() -> str:
    """Get the name of the current environment."""
    if is_colab():
        return "colab"
    elif is_kaggle():
        return "kaggle"
    else:
        return "local"


def verify_cloud_environment(force: bool = False) -> None:
    """
    Verify that we are in a cloud environment.

    Args:
        force: If True, bypass the check and allow local execution.

    Raises:
        RuntimeError: If not in a cloud environment and force is False.
    """
    if not force and not is_cloud_environment():
        raise RuntimeError(
            "Git automation is designed for cloud environments (Kaggle/Colab) to prevent "
            "accidental messy commits during local development. Use --force-auto-push or "
            "force=True to bypass this safety check."
        )
