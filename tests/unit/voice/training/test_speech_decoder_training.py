import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Dataset

from larry.voice.config.criterion.voice_criterion_configs import SpeechDecoderCriterionConfig
from larry.voice.config.data.voice_data_configs import TTSSourceConfig
from larry.voice.criteria.flow_matching import SpeechDecoderCriterion
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedTTSDataset
from larry.voice.data.preprocessing.speaker_mapping import SpeakerMapping
from larry.voice.model.speech_decoder import SpeechDecoderModel
from larry.voice.training.voice_trainers import SpeechDecoderTrainer
from larry.voice.training.voice_viz_callbacks import SpeechDecoderVizCallback
from tests.helpers.audio import wav
from tests.helpers.common import uninitialized
from tests.unit.voice.model.test_speech_decoder import tiny_config


class RecordingMetrics:
    def __init__(self) -> None:
        self.figures: list[str] = []
        self.scalars: dict[str, float] = {}

    def add_figure(self, tag: str, figure: Any, step: int, **kwargs: Any) -> None:
        self.figures.append(tag)

    def add_scalar(self, tag: str, value: float, step: int, **kwargs: Any) -> None:
        self.scalars[tag] = value


def _source(name: str, seconds: list[float], with_f0: bool = False) -> Dataset:
    columns: dict[str, list[Any]] = {
        "id": [f"{name}{i}" for i in range(len(seconds))],
        "audio": [{"bytes": wav(s, 16_000, frequency=220 + 70 * i), "path": None} for i, s in enumerate(seconds)],
        "unit_ids": [[(i * 7 + k) % 81 for k in range(int(s * 25))] for i, s in enumerate(seconds)],
        "speaker_id": [f"s{i % 2}" for i in range(len(seconds))],
        "duration_s": seconds,
    }
    if with_f0:
        frames = [int(s * 16_000) // 320 + 1 for s in seconds]
        columns["f0"] = [[0.0 if k % 5 == 0 else 180.0 + 10 * (k % 3) for k in range(n)] for n in frames]
        columns["aperiodicity"] = [[[0.2] * 8 for _ in range(n)] for n in frames]
    return Dataset.from_dict(columns)


def _trainer(with_f0: bool = False) -> tuple[SpeechDecoderTrainer, RecordingMetrics]:
    torch.manual_seed(0)
    configs = [TTSSourceConfig(name="a", index_path="a"), TTSSourceConfig(name="b", index_path="b")]
    mapping = SpeakerMapping()
    for config in configs:
        for speaker in ("s0", "s1"):
            mapping.add_item(SpeakerMapping.key(config, speaker))
    sources = [_source("a", [1.2, 0.6], with_f0), _source("b", [0.9], with_f0)]
    dataset = IndexedTTSDataset(configs, sources, [None, None], 16_000, mapping)
    metrics = RecordingMetrics()
    model = SpeechDecoderModel(tiny_config(n_speakers=len(mapping.items)))
    trainer = uninitialized(
        SpeechDecoderTrainer,
        _model=model,
        _config=SimpleNamespace(mixture_config=SimpleNamespace(eval_sources=configs)),
        unwrapped_model=model,
        criterion=SpeechDecoderCriterion(SpeechDecoderCriterionConfig()),
        metrics=metrics,
        eval_loader=SimpleNamespace(dataset=dataset),
        train_state=SimpleNamespace(global_step=0),
        _pending_tokens=0,
        accelerator=SimpleNamespace(
            is_main_process=True,
            device=torch.device("cpu"),
            unwrap_model=lambda m: m,
            autocast=contextlib.nullcontext,
        ),
    )
    return trainer, metrics


def _eval_batch(trainer: SpeechDecoderTrainer) -> dict[str, torch.Tensor]:
    dataset = trainer.eval_loader.dataset
    return trainer.make_collate_fn()([dataset[k] for k in range(len(dataset))])


def test_eval_loss_terms_are_per_row_and_grouped_by_source() -> None:
    trainer, _ = _trainer()
    trainer.model.eval()
    batch = _eval_batch(trainer)

    with torch.no_grad():
        output = trainer.eval_forward_pass(batch)

    model_output = output.model_output
    terms = output.terms
    assert set(terms) == {"flow", "prior", "kl"}
    assert all(term.sums.shape == (3,) for term in terms.values())
    elements = model_output.mask.sum(dim=1) * 80
    expected_flow = trainer.criterion.sample_losses(model_output, model_output.targets) / elements
    assert torch.allclose(terms["flow"].sums, expected_flow)
    assert terms["prior"].counts.tolist() == (model_output.frame_mask.sum(dim=1) * 80).tolist()
    assert (terms["flow"].weight, terms["prior"].weight, terms["kl"].weight) == (1.0, 1.0, 0.0)
    assert [ids.tolist() for ids in trainer.eval_group_ids(batch)] == [[0, 0, 1]]
    assert torch.isfinite(output.loss)


def test_training_loss_includes_the_kl_term_after_warmup() -> None:
    trainer, _ = _trainer()
    trainer.model.train()
    batch = _eval_batch(trainer)
    trainer.train_state.global_step = 10_000

    step = trainer._forward_pass(batch)

    output, loss = step.model_output, step.loss
    criterion = trainer.criterion
    prior = criterion.config.prior_weight * criterion.prior_loss(output)
    kl = criterion.config.kl_weight * output.kl.mean()
    assert float(loss) == pytest.approx(float(criterion.flow_loss(output) + prior + kl))


def test_tokens_are_counted_at_mel_rate() -> None:
    trainer, _ = _trainer()
    batch = _eval_batch(trainer)

    trainer.count_tokens(batch)

    assert trainer._pending_tokens == int(batch["unit_id_lengths"].sum()) * 4


def test_viz_logs_a_comparison_per_example_and_restores_training_mode() -> None:
    trainer, metrics = _trainer()
    trainer.model.train()
    callback = SpeechDecoderVizCallback(trainer, viz_steps=10, generation_steps=2)

    callback.viz(10)

    assert sorted(metrics.figures) == [
        "viz/0/a/no_f0/comparison", "viz/1/a/no_f0/comparison", "viz/2/b/no_f0/comparison",
    ]
    assert {"viz/0/a/no_f0/mel_l1", "viz/no_f0/mel_l1"} <= set(metrics.scalars)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.scalars.values())
    assert trainer.model.training


def test_viz_renders_true_flat_and_no_f0_versions_when_the_data_has_f0() -> None:
    trainer, metrics = _trainer(with_f0=True)
    callback = SpeechDecoderVizCallback(trainer, viz_steps=10, generation_steps=2)

    callback.viz(10)

    assert {f"viz/{name}/mel_l1" for name in ("true_f0", "no_f0", "flat_f0")} <= set(metrics.scalars)
    assert "viz/2/b/flat_f0/comparison" in metrics.figures
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.scalars.values())


def test_training_with_f0_adds_the_range_term_and_uses_true_f0_only_during_warmup() -> None:
    trainer, _ = _trainer(with_f0=True)
    trainer.model.train()
    batch = _eval_batch(trainer)
    seen: list[bool] = []
    forward = trainer.model.forward
    trainer.model.forward = lambda *a, **k: seen.append(k["use_true_f0"]) or forward(*a, **k)

    output = trainer._forward_pass(batch)
    trainer.train_state.global_step = trainer.unwrapped_model.config.true_f0_steps

    trainer._forward_pass(batch)
    assert "f0_range" in (output.terms or {})
    assert seen == [True, False]
