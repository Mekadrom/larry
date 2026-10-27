from types import SimpleNamespace

import pytest
from accelerate.utils import DDPCommunicationHookType, DistributedDataParallelKwargs, GradScalerKwargs

from larry.multimodal.training.trainers import LarryMultimodalTrainer
from larry.voice.training.voice_trainers import VocoderTrainer
from tests.helpers.common import uninitialized
from tests.helpers.config_recipes import load_training


def _trainer(cls: type, hook: str, state: dict | None = None) -> object:
    return uninitialized(cls, _config=SimpleNamespace(ddp_comm_hook=hook, ddp_comm_state_option=state or {}))


def test_no_hook_leaves_the_trainers_handlers_alone() -> None:
    assert _trainer(VocoderTrainer, "no").kwargs_handlers() == []

    handlers = _trainer(LarryMultimodalTrainer, "no").kwargs_handlers()
    assert len(handlers) == 1 and handlers[0].comm_hook == DDPCommunicationHookType.NO


def test_a_hook_adds_a_ddp_handler_when_the_trainer_has_none() -> None:
    (handler,) = _trainer(VocoderTrainer, "bf16").kwargs_handlers()

    assert isinstance(handler, DistributedDataParallelKwargs)
    assert handler.comm_hook == DDPCommunicationHookType.BF16


def test_a_hook_merges_into_the_trainers_own_ddp_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer(LarryMultimodalTrainer, "power_sgd", {"matrix_approximation_rank": 2})
    own = trainer.accelerator_kwargs_handlers()
    monkeypatch.setattr(trainer, "accelerator_kwargs_handlers", lambda: [*own, GradScalerKwargs()])

    handlers = trainer.kwargs_handlers()

    ddp = [h for h in handlers if isinstance(h, DistributedDataParallelKwargs)]
    assert len(ddp) == 1 and len(handlers) == 2
    assert ddp[0].find_unused_parameters is True
    assert ddp[0].comm_hook == DDPCommunicationHookType.POWER_SGD
    assert ddp[0].comm_state_option == {"matrix_approximation_rank": 2}


def test_the_overlay_turns_on_bf16_gradient_compression() -> None:
    assert load_training("larry-speech-encoder+ddp_bf16_comm").trainer.ddp_comm_hook == "bf16"
    assert load_training("larry-speech-encoder").trainer.ddp_comm_hook == "no"
