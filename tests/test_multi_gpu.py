import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from src.training.multi_gpu import (
    setup_multi_gpu,
    get_model_state_dict,
    load_model_state_dict,
    calculate_total_batch_size,
    detect_kaggle_multi_gpu,
    print_gpu_info,
)


def test_single_gpu_detection():
    """Single GPU/CPU detection should return the model unwrapped."""
    model = nn.Linear(10, 10)

    # Mocking torch.cuda to simulate CPU only
    with patch("src.training.multi_gpu.torch.cuda.is_available", return_value=False):
        result = setup_multi_gpu(model)
        assert not isinstance(result, nn.DataParallel)

    # Mocking 1 GPU
    with patch("src.training.multi_gpu.torch.cuda.is_available", return_value=True):
        with patch("src.training.multi_gpu.torch.cuda.device_count", return_value=1):
            with patch.object(model, "to", return_value=model):
                result = setup_multi_gpu(model)
                assert not isinstance(result, nn.DataParallel)


def test_multi_gpu_wrapping():
    """Should wrap model in DataParallel when 2+ GPUs are available."""
    model = nn.Linear(10, 10)

    with patch("src.training.multi_gpu.torch.cuda.is_available", return_value=True):
        with patch("src.training.multi_gpu.torch.cuda.device_count", return_value=2):
            with patch("src.training.multi_gpu.print_gpu_info"):
                with patch.object(model, "to", return_value=model):
                    fake_dp = MagicMock(spec=nn.DataParallel)
                    fake_dp.device_ids = [0, 1]
                    with patch(
                        "src.training.multi_gpu.nn.DataParallel", return_value=fake_dp
                    ):
                        result = setup_multi_gpu(model)
                        assert result is fake_dp
                        assert result.device_ids == [0, 1]


def test_state_dict_unwrapping():
    """get_model_state_dict should always return an unwrapped state dict."""
    model = nn.Linear(10, 10)

    # 1. Unwrapped model
    sd_unwrapped = get_model_state_dict(model)
    assert "weight" in sd_unwrapped
    assert "module.weight" not in sd_unwrapped

    # 2. Wrapped model (Mock DataParallel if not on CUDA)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            wrapped = nn.DataParallel(model.cuda())
            sd_wrapped = get_model_state_dict(wrapped)
            assert "module" not in sd_wrapped.keys()
            assert "weight" in sd_wrapped
    else:
        # Just mock a DataParallel structure for CPU testing
        wrapped = nn.DataParallel(
            model
        )  # Note: DataParallel on CPU is deprecated but works for dummy test
        sd_wrapped = get_model_state_dict(wrapped)
        assert "weight" in sd_wrapped


def test_checkpoint_compatibility():
    """load_model_state_dict should work seamlessly with or without DataParallel."""
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)

    # Ensure they start different
    model1.weight.data.fill_(1.0)
    model2.weight.data.fill_(0.0)

    sd = get_model_state_dict(model1)
    load_model_state_dict(model2, sd)

    assert torch.allclose(model1.weight, model2.weight)


def test_batch_size_calculation():
    """calculate_total_batch_size should multiply args."""
    total = calculate_total_batch_size(batch_size_per_gpu=32, num_gpus=2)
    assert total == 64


def test_gpu_info_printing_no_error():
    """print_gpu_info should just not crash."""
    try:
        print_gpu_info()
    except Exception as e:
        assert False, f"print_gpu_info raised an exception {e}"


def test_detect_kaggle_environment():
    """Should correctly identify Kaggle environment."""
    with (
        patch("os.path.exists", return_value=False),
        patch.dict("os.environ", {}, clear=True),
    ):
        assert not detect_kaggle_multi_gpu()

    with (
        patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, clear=True),
        patch("src.training.multi_gpu.torch.cuda.is_available", return_value=True),
        patch("src.training.multi_gpu.torch.cuda.device_count", return_value=2),
    ):
        assert detect_kaggle_multi_gpu()
