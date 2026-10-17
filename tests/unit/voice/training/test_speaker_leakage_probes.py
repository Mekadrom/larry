from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.config.training.voice_trainer_configs import SpeakerLeakageProbesConfig, SpeechTokenizerTrainerConfig
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel
from larry.voice.training.voice_callbacks import PooledActivations, SpeakerLeakageProbesCallback
from larry.voice.training.voice_trainers import SpeechTokenizerTrainer


def _callback(**kwargs: Any) -> SpeakerLeakageProbesCallback:
    return SpeakerLeakageProbesCallback(SpeakerLeakageProbesConfig(**{"probe_every": 100, **kwargs}))


def _keys(counts: dict[str, int]) -> list[str]:
    keys = [key for key, n in counts.items() for _ in range(n)]
    return keys[::2] + keys[1::2]


def _trainer(max_batch_seconds: float) -> SpeechTokenizerTrainer:
    trainer = SpeechTokenizerTrainer.__new__(SpeechTokenizerTrainer)
    vars(trainer).update(_config=SimpleNamespace(mixture_config=SimpleNamespace(max_batch_seconds=max_batch_seconds)))
    return trainer


def test_speakers_below_the_clip_floor_are_dropped() -> None:
    keys = _keys({"a": 3, "b": 2, "c": 5})

    rows, labels = _callback(min_clips=3, max_clips=100)._select_rows(keys)

    assert sorted({keys[r] for r in rows}) == ["a", "c"]
    assert len(rows) == 8


def test_speakers_are_capped_at_max_clips() -> None:
    keys = _keys({"a": 50, "b": 4})

    rows, labels = _callback(min_clips=1, max_clips=10)._select_rows(keys)

    assert torch.bincount(labels).tolist() == [10, 4]


def test_labels_are_dense_sorted_by_key_and_match_each_row() -> None:
    keys = _keys({"z": 4, "m": 4, "a": 4})

    rows, labels = _callback(min_clips=1, max_clips=10)._select_rows(keys)

    names = sorted(set(keys))
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    assert all(names[label] == keys[row] for row, label in zip(rows, labels.tolist(), strict=True))


def test_rows_are_unique() -> None:
    rows, _ = _callback(min_clips=1, max_clips=5)._select_rows(_keys({"a": 20, "b": 20}))

    assert len(rows) == len(set(rows))


def test_selection_is_deterministic_for_a_seed_and_varies_across_seeds() -> None:
    keys = _keys({f"s{i}": 30 for i in range(5)})

    first = _callback(min_clips=1, max_clips=5, seed=1)._select_rows(keys)
    again = _callback(min_clips=1, max_clips=5, seed=1)._select_rows(keys)
    other = _callback(min_clips=1, max_clips=5, seed=2)._select_rows(keys)

    assert first[0] == again[0] and torch.equal(first[1], again[1])
    assert first[0] != other[0]


def test_no_eligible_speakers_gives_empty_selection() -> None:
    rows, labels = _callback(min_clips=10)._select_rows(_keys({"a": 2}))

    assert rows == []
    assert labels.shape == (0,)


def test_duration_batches_sort_by_duration_and_fit_the_padded_budget() -> None:
    durations = torch.tensor([4.0, 1.0, 3.0, 2.0, 10.0, 1.5], dtype=torch.float64)

    batches = _trainer(max_batch_seconds=6.0).duration_batches(durations)

    assert batches == [[1, 5, 3], [2], [0], [4]]


def test_duration_batches_cover_every_row_once() -> None:
    durations = torch.rand(97, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 20

    batches = _trainer(max_batch_seconds=60.0).duration_batches(durations)
    flat = [row for batch in batches for row in batch]

    assert sorted(flat) == list(range(97))
    assert all(len(b) == 1 or len(b) * float(durations[b].max()) <= 60.0 for b in batches)


class _KeyedRows(Dataset[dict[str, Any]]):
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, k: int) -> dict[str, Any]:
        return {"key": self.keys[k]}


def test_subset_batches_line_up_with_selected_labels() -> None:
    keys = _keys({"a": 6, "b": 6, "c": 2})
    durations = torch.arange(len(keys), dtype=torch.float64).flip(0) + 1
    rows, labels = _callback(min_clips=3, max_clips=4)._select_rows(keys)
    batches = _trainer(max_batch_seconds=20.0).duration_batches(durations[rows])

    loader = DataLoader(Subset(_KeyedRows(keys), rows), batch_sampler=batches, collate_fn=lambda items: items)

    names = sorted({"a", "b"})
    for batch, items in zip(batches, loader, strict=True):
        assert [item["key"] for item in items] == [names[label] for label in labels[batch].tolist()]


def _split_callback(n_speakers: int, **kwargs: Any) -> SpeakerLeakageProbesCallback:
    callback = _callback(**kwargs)
    callback.n_speakers = n_speakers
    return callback


def test_split_is_disjoint_and_covers_every_row() -> None:
    labels = torch.tensor([0] * 10 + [1] * 13 + [2] * 20)[torch.randperm(43, generator=torch.Generator().manual_seed(0))]

    train, test = _split_callback(3)._split_rows(labels)

    assert set(train.tolist()).isdisjoint(test.tolist())
    assert sorted(train.tolist() + test.tolist()) == list(range(43))


def test_split_puts_every_speaker_on_both_sides_at_the_test_fraction() -> None:
    counts = [10, 13, 20, 2]
    labels = torch.cat([torch.full((n,), s) for s, n in enumerate(counts)])

    train, test = _split_callback(len(counts), test_fraction=0.25)._split_rows(labels)

    for speaker, n in enumerate(counts):
        n_test = int((labels[test] == speaker).sum())
        assert int((labels[train] == speaker).sum()) > 0
        assert abs(n_test - n * 0.25) <= 1
        assert n_test >= 1


def test_split_is_deterministic_for_a_seed() -> None:
    labels = torch.arange(40) % 4

    first = _split_callback(4, seed=3)._split_rows(labels)
    again = _split_callback(4, seed=3)._split_rows(labels)

    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(2, 3)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return self.inner(x), torch.zeros(x.shape[0], dtype=torch.long)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()
        self.downsampled = nn.Identity()

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> torch.Tensor:
        frames = waveforms.unsqueeze(-1).expand(-1, -1, 2)
        encoded, _ = self.encoder(frames)
        return self.downsampled(encoded[:, ::2])


def test_pooling_ignores_padding() -> None:
    model = _ToyModel()
    activations = PooledActivations(["encoder"])
    clip = torch.randn(1, 6)
    padded = torch.cat([clip, torch.full((1, 4), 100.0)], dim=1)

    with activations.attached(model):
        activations.begin_batch(torch.tensor([6]), 6)
        model(clip, torch.tensor([6]))
        activations.begin_batch(torch.tensor([6]), 10)
        model(padded, torch.tensor([6]))
    alone, in_padding = activations.features()["encoder"]

    assert torch.allclose(alone, in_padding, atol=1e-5)


def test_pooled_features_are_mean_then_std_of_the_first_float_output() -> None:
    model = _ToyModel()
    activations = PooledActivations(["encoder"])
    waveforms = torch.randn(2, 5)

    with activations.attached(model):
        activations.begin_batch(torch.tensor([5, 5]), 5)
        model(waveforms, torch.tensor([5, 5]))
    pooled = activations.features()["encoder"]

    encoded = model.encoder.inner(waveforms.unsqueeze(-1).expand(-1, -1, 2))
    assert pooled.shape == (2, 6)
    assert torch.allclose(pooled[:, :3], encoded.mean(dim=1), atol=1e-5)
    assert torch.allclose(pooled[:, 3:], encoded.std(dim=1, unbiased=False), atol=1e-5)


def test_each_layer_masks_at_its_own_frame_rate() -> None:
    model = _ToyModel()
    activations = PooledActivations(["encoder", "downsampled"])
    waveforms = torch.cat([torch.randn(1, 4), torch.full((1, 4), 100.0)], dim=1)

    with activations.attached(model):
        activations.begin_batch(torch.tensor([4]), 8)
        model(waveforms, torch.tensor([4]))
    features = activations.features()

    encoded = model.encoder.inner(waveforms.unsqueeze(-1).expand(-1, -1, 2))
    assert torch.allclose(features["encoder"][0, :3], encoded[0, :4].mean(dim=0), atol=1e-5)
    assert torch.allclose(features["downsampled"][0, :3], encoded[0, 0:4:2].mean(dim=0), atol=1e-5)


def test_hooks_are_removed_after_the_block_even_when_the_forward_raises() -> None:
    model = _ToyModel()
    activations = PooledActivations(["encoder"])

    with pytest.raises(RuntimeError), activations.attached(model):
        activations.begin_batch(torch.tensor([3]), 3)
        raise RuntimeError("forward failed")

    model(torch.randn(1, 3), torch.tensor([3]))
    assert activations.pooled["encoder"] == []
    assert not model.encoder._forward_hooks


def test_unknown_layer_names_fail_before_any_forward() -> None:
    with pytest.raises(ValueError, match="no_such_layer"), PooledActivations(["no_such_layer"]).attached(_ToyModel()):
        pass


def test_non_sequence_outputs_are_rejected() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    activations = PooledActivations(["0"])

    with pytest.raises(ValueError, match=r"\[B, T, D\]"), activations.attached(model):
        activations.begin_batch(torch.tensor([4]), 4)
        model(torch.randn(2, 4))


def _fake_trainer(model: nn.Module) -> Any:
    return SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda m: m, device=torch.device("cpu"), is_main_process=True),
    )


def test_extraction_restores_training_mode_and_follows_loader_order() -> None:
    model = _ToyModel()
    model.train()
    callback = _callback(layers=["encoder"])
    waveforms = [torch.randn(1, 4), torch.randn(1, 6)]
    callback.loader = cast(DataLoader, [
        {"waveforms": w, "waveform_lengths": torch.tensor([w.shape[1]])} for w in waveforms
    ])

    features = callback._extract_features(cast(SpeechTokenizerTrainer, _fake_trainer(model)))

    assert model.training
    assert features["encoder"].shape == (2, 6)
    assert not model.encoder._forward_hooks
    for row, w in enumerate(waveforms):
        encoded = model.encoder.inner(w.unsqueeze(-1).expand(-1, -1, 2))
        assert torch.allclose(features["encoder"][row, :3], encoded[0].mean(dim=0), atol=1e-5)


def _probe_callback(labels: torch.Tensor, **kwargs: Any) -> SpeakerLeakageProbesCallback:
    callback = _callback(**kwargs)
    callback.n_speakers = int(labels.max()) + 1
    callback.labels = labels
    callback.train_rows, callback.test_rows = callback._split_rows(labels)
    return callback


def test_separable_speakers_are_recovered_on_held_out_rows() -> None:
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(400) % 20
    centers = torch.randn(20, 32, generator=generator) * 3
    features = centers[labels] + torch.randn(400, 32, generator=generator)
    callback = _probe_callback(labels, train_epochs=200, mlp_inner_dim=64)

    accuracies = callback._probe_accuracies(features, torch.device("cpu"))

    assert accuracies["linear"][1] > 0.9
    assert accuracies["mlp"][1] > 0.9


def test_noise_stays_near_chance_on_held_out_rows_while_train_accuracy_rises() -> None:
    labels = torch.arange(400) % 20
    features = torch.randn(400, 256, generator=torch.Generator().manual_seed(1))
    callback = _probe_callback(labels, train_epochs=200, mlp_inner_dim=256)

    accuracies = callback._probe_accuracies(features, torch.device("cpu"))

    for train_accuracy, test_accuracy in accuracies.values():
        assert test_accuracy < 0.15
        assert train_accuracy > 0.5


def test_probe_training_leaves_the_global_rng_untouched() -> None:
    labels = torch.arange(40) % 4
    callback = _probe_callback(labels, train_epochs=3, mlp_inner_dim=8)
    features = torch.randn(40, 8)
    torch.manual_seed(123)
    expected = torch.rand(3)

    torch.manual_seed(123)
    callback._probe_accuracies(features, torch.device("cpu"))

    assert torch.equal(torch.rand(3), expected)


def test_probe_results_are_reproducible() -> None:
    labels = torch.arange(80) % 8
    features = torch.randn(80, 16, generator=torch.Generator().manual_seed(2))

    first = _probe_callback(labels, train_epochs=20, mlp_inner_dim=16)._probe_accuracies(features, torch.device("cpu"))
    again = _probe_callback(labels, train_epochs=20, mlp_inner_dim=16)._probe_accuracies(features, torch.device("cpu"))

    assert first == again


def test_steps_off_the_schedule_and_non_main_processes_do_nothing() -> None:
    callback = _callback(probe_every=10)
    trainer = SpeechTokenizerTrainer.__new__(SpeechTokenizerTrainer)
    vars(trainer).update(accelerator=SimpleNamespace(is_main_process=False))

    assert callback.on_step_end(trainer, torch.tensor(0.0), 7) is None
    assert callback.on_step_end(trainer, torch.tensor(0.0), 10) is None
    assert callback.loader is None


def test_trainer_config_builds_the_probes_config_from_overrides() -> None:
    config = SpeechTokenizerTrainerConfig(
        max_steps=1, tokenizer_model_path="t", speaker_mapping_path="s",
        speaker_leakage_steps=500, speaker_leakage_probes_callback_overrides={"train_epochs": 7, "layers": ["fsq"]},
    )
    disabled = SpeechTokenizerTrainerConfig(max_steps=1, tokenizer_model_path="t", speaker_mapping_path="s")

    assert config.speaker_leakage_probes_config == SpeakerLeakageProbesConfig(
        probe_every=500, train_epochs=7, layers=["fsq"]
    )
    assert disabled.speaker_leakage_probes_config is None


def test_default_layers_exist_in_the_speech_tokenizer_and_output_sequences() -> None:
    model_config = SpeechTokenizerModelConfig(
        d_model=128, n_heads=2, n_query_groups=2, n_encoder1_layers=1, n_encoder2_layers=1, text_vocab_size=8
    )
    model = SpeechTokenizerModel(model_config).eval()
    activations = PooledActivations(SpeakerLeakageProbesConfig(probe_every=1).layers)

    with torch.no_grad(), activations.attached(model):
        activations.begin_batch(torch.tensor([16000, 8000]), 16000)
        model(torch.randn(2, 16000), torch.tensor([16000, 8000]))

    for layer, features in activations.features().items():
        assert features.shape == (2, 256), layer
