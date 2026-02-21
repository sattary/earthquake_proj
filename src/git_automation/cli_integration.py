import os
from typing import Optional, Callable

from .callback import AutoPushCallback
from .environment import is_cloud_environment


def add_auto_push_args() -> Callable:
    """
    Decorator for Typer CLI commands to inject standard
    auto-push arguments symmetrically across all entrypoints.
    """

    def decorator(f):
        # We don't statically inject args here due to Typer's introspection,
        # but this decorator serves as a marker/enforcer if we wanted to
        # dynamically modify the signature. For Typer, it's easier to explicitly
        # add the options in the main command function.
        return f

    return decorator


def validate_auto_push_config(interval: int, dry_run: bool, force: bool) -> bool:
    """Validate configuration combinations are sensible."""
    if interval is None or interval <= 0:
        return False

    if not is_cloud_environment() and not force and not dry_run:
        print(
            "[AutoPush Notice] Ignoring auto-push args outside cloud environment. Use --force-auto-push to override."
        )
        return False

    return True


def create_auto_push_callback(
    run_dir: str,
    interval: Optional[int],
    dry_run: bool = False,
    force: bool = False,
    pat: Optional[str] = None,
    include_checkpoints: bool = True,
) -> Optional[AutoPushCallback]:
    """
    Factory function to easily create the callback from CLI arguments.
    Returns None if validation fails or interval is not set.
    """
    if not interval or interval <= 0:
        return None

    if not validate_auto_push_config(interval, dry_run, force):
        return None

    # Resolve PAT from env var if not provided specifically via CLI
    final_pat = pat or os.environ.get("GITHUB_PAT")

    # If not local and no PAT (and not dry running), warn
    if is_cloud_environment() and not dry_run and not final_pat:
        print(
            "[AutoPush Warning] Trying to run auto-push in cloud without a GitHub PAT."
        )
        print(
            "Commits will be created but they cannot be pushed. Set GITHUB_PAT env var first."
        )

    return AutoPushCallback(
        run_dir=run_dir,
        push_interval=interval,
        pat=final_pat,
        dry_run=dry_run,
        force=force,
        include_checkpoints=include_checkpoints,
    )
