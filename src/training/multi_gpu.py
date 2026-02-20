import os
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any


def print_gpu_info() -> None:
    """Print information about available GPUs."""
    if not torch.cuda.is_available():
        print("[gpu] No GPUs detected. Running on CPU.")
        return

    count = torch.cuda.device_count()
    print(f"[gpu] {count} GPU(s) detected:")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {mem_gb:.2f} GB")
        print(f"    Compute: {props.major}.{props.minor}")


def detect_kaggle_multi_gpu() -> bool:
    """Detect if running on Kaggle with multiple GPUs."""
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    return is_kaggle and torch.cuda.is_available() and torch.cuda.device_count() >= 2


def setup_multi_gpu(model: nn.Module, gpu_ids: Optional[List[int]] = None) -> nn.Module:
    """
    Wrap model in DataParallel if multiple GPUs available.

    Args:
        model: nn.Module to wrap
        gpu_ids: List of GPU IDs to use. If None, uses all available.

    Returns:
        DataParallel-wrapped model (if 2+ GPUs) or original model
    """
    if not torch.cuda.is_available():
        return model

    count = torch.cuda.device_count()
    if count <= 1:
        # Move to cuda:0 if not already
        return model.to(torch.device("cuda:0"))

    if gpu_ids is not None:
        # Validate requested IDs
        valid_ids = [i for i in gpu_ids if 0 <= i < count]
        if not valid_ids:
            print(
                f"[gpu] Warning: Requested GPU IDs {gpu_ids} are invalid. Using all available."
            )
            valid_ids = list(range(count))
        gpu_ids = valid_ids
    else:
        gpu_ids = list(range(count))

    print_gpu_info()
    print(f"[gpu] Wrapping model in DataParallel using devices: {gpu_ids}")
    return nn.DataParallel(model.to(torch.device("cuda")), device_ids=gpu_ids)


def get_model_state_dict(model: nn.Module) -> Dict[str, Any]:
    """
    Extract state dict, unwrapping DataParallel if needed.
    Always use this instead of model.state_dict() for checkpoint saving.
    """
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    """
    Load state dict, handling DataParallel wrapper.
    """
    # strict=False is often useful for custom architectures, but we'll stick to
    # the requested behavior. If the prompt requires specific strict handling, it
    # can be passed down, but for now we default to what pytorch does.
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)


def calculate_total_batch_size(batch_size_per_gpu: int, num_gpus: int) -> int:
    """Calculate effective batch size."""
    return batch_size_per_gpu * num_gpus
