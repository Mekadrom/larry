from types import SimpleNamespace

import pytest
import torch
from torch import nn

from larry.common.config.training.trainer_configs import CompileConfig
from larry.common.model.compile import ModelCompiler
from larry.multimodal.training.trainers import LarryMultimodalTrainer
from tests.helpers.common import uninitialized
from tests.helpers.config_recipes import load_training
from tests.unit.common.model.test_recurrent_trunk import _trunk, _x


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        self.inner = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.inner:
            x = layer(x)
        return self.mlp(x)


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = Block()
        self.stack = nn.ModuleList([Block(), Block()])
        self.shared = nn.ModuleList([shared, shared, shared])
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in [*self.stack, *self.shared]:
            x = block(x)
        return self.head(x)


@pytest.fixture
def compiled(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []

    def fake_compile(fn: object, **kwargs: object) -> object:
        calls.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


def test_blocks_mode_compiles_each_outermost_repeated_block_once(compiled: list[object]) -> None:
    model = Toy()
    keys = list(model.state_dict())

    count = ModelCompiler(CompileConfig(mode="blocks")).apply(model)

    assert count == 3 == len(compiled)
    assert {fn.__self__ for fn in compiled} == {model.stack[0], model.stack[1], model.shared[0]}
    assert list(model.state_dict()) == keys and not any("_orig_mod" in k for k in model.state_dict())


def test_model_mode_compiles_only_the_top_level_forward(compiled: list[object]) -> None:
    model = Toy()

    assert ModelCompiler(CompileConfig(mode="model")).apply(model) == 1
    assert compiled[0].__self__ is model


def test_compiling_is_idempotent_and_off_by_default(compiled: list[object]) -> None:
    model = Toy()
    compiler = ModelCompiler(CompileConfig(mode="blocks"))

    compiler.apply(model)
    compiler.apply(model)

    assert len(compiled) == 3
    assert ModelCompiler(CompileConfig()).apply(Toy()) == 0


def test_the_recurrent_trunk_compiles_per_block_and_keeps_its_loop_eager(compiled: list[object]) -> None:
    trunk = _trunk()

    ModelCompiler(CompileConfig(mode="blocks")).apply(trunk)

    assert {fn.__self__ for fn in compiled} == set(trunk.blocks)
    assert "forward" not in trunk.__dict__


@pytest.mark.parametrize("checkpointing", [False, True])
def test_a_compiled_trunk_matches_eager_through_truncated_backprop(checkpointing: bool) -> None:
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    eager, compiled_trunk = _trunk(gradient_checkpointing=checkpointing), _trunk(gradient_checkpointing=checkpointing)
    ModelCompiler(CompileConfig(mode="blocks", backend="aot_eager")).apply(compiled_trunk)

    for _ in range(3):
        outputs = []
        for trunk in (eager, compiled_trunk):
            trunk.train()
            x = _x(7).requires_grad_()
            output = trunk(x).logits
            output.square().mean().backward()
            outputs.append((output.detach(), x.grad, [p.grad.clone() for p in trunk.parameters() if p.grad is not None]))
            trunk.zero_grad(set_to_none=True)

        (eager_out, eager_x_grad, eager_grads), (out, x_grad, grads) = outputs
        assert torch.allclose(out, eager_out, atol=1e-5)
        assert torch.allclose(x_grad, eager_x_grad, atol=1e-5)
        assert len(grads) == len(eager_grads) and all(
            torch.allclose(a, b, atol=1e-5) for a, b in zip(grads, eager_grads, strict=True)
        )

    assert 1 <= counters["stats"]["unique_graphs"] <= 4 * len(compiled_trunk.blocks)


def test_the_multimodal_trainer_refuses_whole_model_compilation() -> None:
    trainer = uninitialized(LarryMultimodalTrainer, _config=SimpleNamespace(compile=CompileConfig(mode="model")))

    with pytest.raises(ValueError, match="compile.mode=blocks"):
        trainer.compile_model(Toy())


def test_the_overlay_turns_on_block_compilation() -> None:
    assert load_training("larry-speech-encoder+compile_blocks").trainer.compile.mode == "blocks"
    assert load_training("larry-speech-encoder").trainer.compile.mode == "none"


@pytest.mark.parametrize("training", [False, True])
def test_growing_sequence_lengths_do_not_recompile_compiled_blocks(training: bool) -> None:
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    trunk = _trunk(mean_thinking_steps=2, backprop_depth=1)
    ModelCompiler(CompileConfig(mode="blocks", backend="aot_eager")).apply(trunk)
    trunk.train(training)

    with torch.set_grad_enabled(training):
        for seq_len in range(4, 40, 3):
            trunk(torch.randn(2, seq_len, 32)).logits.sum()

    assert counters["stats"]["unique_graphs"] <= 2 * len(trunk.blocks)


def _fsmn_block(d_model: int = 16) -> nn.Module:
    from larry.voice.config.model.voice_model_configs import SpeechEncoderModelConfig
    from larry.voice.model.speech_encoder import SpeechEncoderEncoderBlock

    torch.manual_seed(0)
    config = SpeechEncoderModelConfig(
        d_model=d_model, n_heads=2, n_query_groups=2, rotary_embedding_dim=8, vocab_size=8,
    )
    return SpeechEncoderEncoderBlock(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_compiled_fsmn_memory_matches_the_depthwise_conv(dtype: torch.dtype) -> None:
    attention = _fsmn_block().attn
    x = torch.randn(2, 16, 37).to(dtype)

    expected = attention.fsmn.to(dtype)(x)
    shifted = attention._shifted_depthwise(x)

    assert shifted.shape == expected.shape and shifted.dtype == expected.dtype
    tolerance = 1e-5 if dtype == torch.float32 else 3e-2
    assert torch.allclose(shifted.float(), expected.float(), atol=tolerance, rtol=tolerance)


def test_an_inductor_compiled_fsmn_block_trains_on_new_lengths_without_recompiling() -> None:
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    block = _fsmn_block().train()
    eager = _fsmn_block().train()
    ModelCompiler(CompileConfig(mode="model")).apply(block)

    for seq_len in (26, 41, 66):
        x = torch.randn(3, seq_len, 16)
        mask = torch.ones(3, seq_len, dtype=torch.bool)
        mask[1, seq_len - 5:] = False
        out = block(x, mask)
        out.sum().backward()
        assert torch.allclose(out, eager(x, mask), atol=1e-4)

    assert counters["stats"]["unique_graphs"] <= 2


def test_toggling_the_qk_probe_needs_only_one_extra_compiled_version() -> None:
    from torch._dynamo.utils import counters

    from larry.common.training.qk_clip import QKClipHandler

    torch._dynamo.reset()
    counters.clear()
    block = _fsmn_block().train()
    assert "qk_probe_active" in block.attn.__dict__
    ModelCompiler(CompileConfig(mode="model", backend="aot_eager")).apply(block)
    handler = QKClipHandler(block, tau=100.0, probe_every=2)

    for step in range(6):
        handler.maybe_arm(step)
        block(torch.randn(2, 12, 16), None).sum().backward()

    assert counters["stats"]["unique_graphs"] <= 2
