import os

# Essential for cloud environments to avoid display connection errors
# Must be set before matplotlib or seaborn imports
os.environ["MPLBACKEND"] = "Agg"

from .callback import AutoPushCallback
from .environment import is_kaggle, is_colab, is_cloud_environment, get_environment_name
from .git_lfs_setup import setup_git_lfs
from .cli_integration import add_auto_push_args, create_auto_push_callback

__all__ = [
    "AutoPushCallback",
    "is_kaggle",
    "is_colab",
    "is_cloud_environment",
    "get_environment_name",
    "setup_git_lfs",
    "add_auto_push_args",
    "create_auto_push_callback",
]
