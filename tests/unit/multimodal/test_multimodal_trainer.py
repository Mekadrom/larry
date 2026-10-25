import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset as HFDataset

from larry.common.config.training.definition import TrainingDefinition, TrainingRunConfig
from larry.common.config.training.training_config import TrainingRunState
from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.common.data.dataloading.weighted_batch_sampler import FixedSizeWeightedBatchSampler
from larry.multimodal.config.data.data_configs import MultimodalMixtureConfig, MultimodalSourceConfig
from larry.multimodal.config.training.trainer_configs import LarryMultimodalTrainerConfig
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.model.multimodal_model import PROCESSOR_DIR, LarryMultimodalModel
from larry.multimodal.processor.processor import LarryMultimodalProcessor
from larry.multimodal.training.trainers import EXPORT_DIR, LarryMultimodalTrainer
from tests.helpers.common import provenance, reset_accelerate_state, spawn_launcher
from tests.helpers.metrics import RecordingMetrics
from tests.unit.multimodal.test_multimodal_generation import _image, _vocab, _voice
from tests.unit.multimodal.test_multimodal_model import _config

ROWS = {
    "voice_text": [{"text": "say this", "unit_ids": [4, 5, 6]}, {"text": "hello there", "unit_ids": [1, 2]}],
    "text_voice": [{"text": "say this", "unit_ids": [7, 8]}, {"text": "a cat", "unit_ids": [3, 4, 5]}],
    "text_image": [{"text": "a cat", "latents": np.ones((3, 2, 4), dtype=np.float32).tolist()}],
    "image_text": [{"text": "a mat", "latents": np.zeros((3, 4, 2), dtype=np.float32).tolist()}],
    "text_only": [{"text": "hello there friend"}, {"text": "say this out loud"}],
}


class ToyMultimodalDataset(IndexedDataset[MultimodalSourceConfig]):
    def get_data(self, source_idx: int, entry: dict[Any, Any]) -> Any:
        return None

    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        config = self.configs[source_idx]
        item = {"text": entry["text"], "is_modal_synthesis": config.is_modal_synthesis, "source": source_idx}
        if entry.get("unit_ids") is not None:
            item["voice"] = np.array(entry["unit_ids"])
        if entry.get("latents") is not None:
            item["image"] = np.array(entry["latents"], dtype=np.float32)
        return item


class ToyMultimodalTrainer(LarryMultimodalTrainer, register=False):
    def make_dataset_dict(self) -> Mapping[str, ToyMultimodalDataset]:
        mixture = self.config.mixture_config
        return {
            "train": self.toy_dataset(mixture.train_sources),
            "validation": self.toy_dataset(mixture.eval_sources),
        }

    @staticmethod
    def toy_dataset(sources: list[MultimodalSourceConfig]) -> ToyMultimodalDataset:
        indexes = [HFDataset.from_list([{"id": str(i), **row} for i, row in enumerate(ROWS[s.task])]) for s in sources]
        return ToyMultimodalDataset(sources, indexes, [None] * len(sources))


def _sources(*tasks: str) -> list[MultimodalSourceConfig]:
    return [MultimodalSourceConfig(name=task, index_path=task, task=task) for task in tasks]


def _trainer_config(collator: PretrainingMultimodalCollator, tasks: list[str], **overrides: Any) -> Any:
    return LarryMultimodalTrainerConfig(**{
        "mixture_config": MultimodalMixtureConfig(
            train_sources=_sources(*tasks), eval_sources=_sources(*tasks), pool_size=16
        ),
        "processor_model_path": collator.config.processor_model_path,
        "image_latent_channels": 3,
        "max_steps": 3,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "num_dataloader_workers": 0,
        "mixed_precision": "no",
        "logging_steps": 1,
        "save_steps": 3,
        "seed": 3,
        **overrides,
    })


def _modes(tasks: list[str]) -> list[str]:
    return sorted({"text"} | {mode for task in tasks for mode in ("voice", "image") if mode in task})


def _template(collator: PretrainingMultimodalCollator, tasks: list[str]) -> Any:
    return _config(_vocab(collator), include_tasks=tasks, include_modes=_modes(tasks), **_voice(1), **_image())


def _trainer(
        collator: PretrainingMultimodalCollator, tmp_path: Path, tasks: list[str], **overrides: Any,
) -> ToyMultimodalTrainer:
    reset_accelerate_state()
    config = _trainer_config(collator, tasks, **overrides)
    model_config = ToyMultimodalTrainer.prepare_model_config(_template(collator, tasks), config)
    torch.manual_seed(0)
    return ToyMultimodalTrainer(provenance(), LarryMultimodalModel(model_config), config, RecordingMetrics(str(tmp_path)))


ALL_TASKS = ["voice_text", "text_voice", "text_image", "image_text", "text_only"]


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_a_mixed_task_run_trains_every_head_and_exports_a_loadable_model(
        collator: PretrainingMultimodalCollator, tmp_path: Path, optimizer_name: str,
) -> None:
    trainer = _trainer(collator, tmp_path, ALL_TASKS, optimizer_name=optimizer_name, eval_steps=3, lr=1e-3)
    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}

    trainer.run()

    model = trainer.accelerator.unwrap_model(trainer.model)
    changed = {name for name, p in model.named_parameters() if not torch.equal(before[name], p)}
    for module in ("text_prelude", "voice_prelude", "voice_coda", "image_prelude", "image_coda", "recurrent_block"):
        assert any(name.startswith(module) for name in changed), module

    export = tmp_path / "checkpoint-3" / EXPORT_DIR
    loaded = LarryMultimodalModel.from_pretrained(export)
    loaded.check_processor(LarryMultimodalProcessor.from_pretrained(export / PROCESSOR_DIR))
    assert loaded.config == model.config
    assert json.loads((tmp_path / "config.json").read_text())["include_tasks"] == ALL_TASKS
    assert trainer.train_state.global_step == 3


def test_muon_leaves_embeddings_and_the_velocity_output_to_adamw(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    trainer = _trainer(collator, tmp_path, ALL_TASKS, optimizer_name="muon")

    muon, adamw = trainer.optimizers
    model = trainer.accelerator.unwrap_model(trainer.model)
    in_muon = {id(p) for group in muon.param_groups for p in group["params"]}
    for name in ("text_prelude.wte.weight", "voice_prelude.ute.weight", "image_coda.row_queries.weight",
                 "image_coda.out.weight"):
        assert id(model.get_parameter(name)) not in in_muon, name
    assert id(model.get_parameter("recurrent_block.blocks.0.self_attn.q_proj.weight")) in in_muon
    assert adamw.param_groups


@pytest.mark.parametrize(("tasks", "absent"), [
    (["voice_text"], {"voice_coda", "image_prelude", "image_coda"}),
    (["text_voice"], {"image_prelude", "image_coda"}),
    (["text_image"], {"voice_prelude", "voice_coda", "image_prelude"}),
])
def test_the_trainer_builds_only_the_modules_its_tasks_use(
        collator: PretrainingMultimodalCollator, tmp_path: Path, tasks: list[str], absent: set[str],
) -> None:
    trainer = _trainer(collator, tmp_path, tasks)

    model = trainer.accelerator.unwrap_model(trainer.model)
    assert all(getattr(model, name) is None for name in absent)
    trainer.run()
    assert trainer.train_state.global_step == 3


def test_every_task_needs_a_train_source_and_sources_must_be_configured_tasks(
        collator: PretrainingMultimodalCollator,
) -> None:
    template = _template(collator, ["voice_text", "text_voice"])

    with pytest.raises(ValueError, match="without a train source"):
        LarryMultimodalTrainer.prepare_model_config(template, _trainer_config(collator, ["voice_text"]))
    with pytest.raises(ValueError, match="outside include_tasks"):
        LarryMultimodalTrainer.prepare_model_config(
            template, _trainer_config(collator, ["voice_text", "text_voice", "text_only"])
        )


def test_the_voice_vocab_comes_from_the_speech_encoders_config(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"fsq_dims": 2, "fsq_levels": 5}))
    config = _trainer_config(collator, ["text_voice"], speech_encoder_model_path=str(tmp_path))

    built = LarryMultimodalTrainer.prepare_model_config(_template(collator, ["text_voice"]), config)

    assert (built.voice_prelude_config.vocab_size, built.voice_coda_config.vocab_size) == (25, 26)


def test_the_sampler_mixes_sources_by_weight_and_resumes() -> None:
    source_ids = torch.tensor([0] * 50 + [1] * 50)
    sampler = FixedSizeWeightedBatchSampler(source_ids, [3.0, 1.0], batch_size=4, pool_size=400, seed=1)

    batches = [batch for batch, _ in zip(sampler, range(100), strict=False)]
    resumed = FixedSizeWeightedBatchSampler(source_ids, [3.0, 1.0], batch_size=4, pool_size=400, seed=1)
    resumed.start_at(40)

    rows = torch.tensor([row for batch in batches for row in batch])
    assert all(len(batch) == 4 for batch in batches)
    assert float((rows < 50).float().mean()) == pytest.approx(0.75, abs=0.06)
    assert any(len({row < 50 for row in batch}) == 2 for batch in batches)
    assert [batch for batch, _ in zip(resumed, range(10), strict=False)] == batches[40:50]


def test_the_run_state_builds_the_model_from_the_trainers_derived_config(
        collator: PretrainingMultimodalCollator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_accelerate_state()
    def toy_datasets(trainer: LarryMultimodalTrainer) -> Mapping[str, ToyMultimodalDataset]:
        mixture = trainer.config.mixture_config
        return {
            "train": ToyMultimodalTrainer.toy_dataset(mixture.train_sources),
            "validation": ToyMultimodalTrainer.toy_dataset(mixture.eval_sources),
        }

    monkeypatch.setattr(LarryMultimodalTrainer, "make_dataset_dict", toy_datasets)
    definition = TrainingDefinition(
        training=TrainingRunConfig(metrics_backend="tensorboard"),
        trainer=_trainer_config(collator, ["voice_text"]),
        model=_template(collator, ["voice_text"]),
    )

    state = TrainingRunState(definition, tmp_path)

    model = state.trainer.accelerator.unwrap_model(state.trainer.model)
    assert type(state.trainer) is LarryMultimodalTrainer
    assert model.voice_coda is None and model.image_prelude is None


def _train_on_two_processes(processor_model_path: str, log_dir: str) -> None:
    from larry.multimodal.config.data.dataloading.dataloading_configs import PretrainingMultimodalCollatorConfig

    config = PretrainingMultimodalCollatorConfig(processor_model_path=processor_model_path)
    collator = PretrainingMultimodalCollator(config)
    trainer = _trainer(collator, Path(log_dir), ["text_voice", "text_only"], train_batch_size=1, max_steps=4)
    trainer.run()
    assert trainer.train_state.global_step == 4


def test_ranks_whose_batches_skip_a_modality_still_train_together(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    spawn_launcher(_train_on_two_processes, args=(collator.config.processor_model_path, str(tmp_path)))
