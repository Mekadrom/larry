from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from larry.voice.training.voice_trainers import VocoderTrainer
from tests.helpers.common import uninitialized
from tests.helpers.config_recipes import load_training


@pytest.fixture(autouse=True)
def restore_backends() -> Iterator[None]:
    saved = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.get_float32_matmul_precision()
    yield
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[:2]
    torch.set_float32_matmul_precision(saved[2])


def _configure(**settings: object) -> None:
    config = {"cudnn_benchmark": False, "cudnn_deterministic": True, "float32_matmul_precision": "highest", **settings}
    uninitialized(VocoderTrainer, _config=SimpleNamespace(**config)).configure_backends()


def test_the_defaults_keep_deterministic_cudnn_and_full_fp32_matmuls() -> None:
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False

    _configure()

    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.get_float32_matmul_precision() == "highest"


def test_each_setting_is_applied() -> None:
    _configure(cudnn_benchmark=True, cudnn_deterministic=False, float32_matmul_precision="high")

    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.get_float32_matmul_precision() == "high"


@pytest.mark.parametrize(("overlay", "field", "value"), [
    ("cudnn_benchmark", "cudnn_benchmark", True),
    ("cudnn_nondeterministic", "cudnn_deterministic", False),
    ("tf32_matmul", "float32_matmul_precision", "high"),
])
def test_the_overlays_set_their_field(overlay: str, field: str, value: object) -> None:
    assert getattr(load_training(f"larry-speech-encoder+{overlay}").trainer, field) == value
    assert getattr(load_training("larry-speech-encoder").trainer, field) != value
