import json
from pathlib import Path
from typing import Any

import pytest
import torch

from larry.common.config.training.definition import TrainingDefinition, TrainingRunConfig
from larry.common.config.training.training_config import TrainingRunState
from larry.multimodal.config.data.data_configs import MultimodalMixtureConfig
from larry.multimodal.config.training.trainer_configs import LarryMultimodalTrainerConfig
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.model.multimodal_model import PROCESSOR_DIR, LarryMultimodalModel
from larry.multimodal.processor.processor import LarryMultimodalProcessor
from larry.multimodal.training.trainers import EXPORT_DIR, LarryMultimodalTrainer
from tests.helpers.common import provenance, reset_accelerate_state, spawn_launcher
from tests.helpers.metrics import RecordingMetrics
from tests.unit.multimodal.test_indexed_multimodal_dataset import source_configs, write_sources
from tests.unit.multimodal.test_multimodal_generation import _image, _vocab, _voice
from tests.unit.multimodal.test_multimodal_model import _config


def _trainer_config(
        collator: PretrainingMultimodalCollator,
        root: Path,
        tasks: list[str],
        max_batch_tokens: int = 48,
        written: bool = False,
        **overrides: Any,
) -> Any:
    # spawned ranks share data the parent already wrote; rewriting it would race the other rank's reads
    sources = source_configs(root / "data", tasks) if written else write_sources(root / "data", tasks)
    return LarryMultimodalTrainerConfig(**{
        "mixture_config": MultimodalMixtureConfig(
            train_sources=sources, eval_sources=sources, pool_size=16, max_batch_tokens=max_batch_tokens
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
) -> LarryMultimodalTrainer:
    reset_accelerate_state()
    config = _trainer_config(collator, tmp_path, tasks, **overrides)
    model_config = LarryMultimodalTrainer.prepare_model_config(_template(collator, tasks), config)
    torch.manual_seed(0)
    model = LarryMultimodalModel(model_config)
    return LarryMultimodalTrainer(provenance(), model, config, RecordingMetrics(str(tmp_path / "run")))


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

    export = tmp_path / "run" / "checkpoint-3" / EXPORT_DIR
    loaded = LarryMultimodalModel.from_pretrained(export)
    loaded.check_processor(LarryMultimodalProcessor.from_pretrained(export / PROCESSOR_DIR))
    assert loaded.config == model.config
    assert json.loads((tmp_path / "run" / "config.json").read_text())["include_tasks"] == ALL_TASKS
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
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    template = _template(collator, ["voice_text", "text_voice"])

    with pytest.raises(ValueError, match="without a train source"):
        LarryMultimodalTrainer.prepare_model_config(template, _trainer_config(collator, tmp_path, ["voice_text"]))
    with pytest.raises(ValueError, match="outside include_tasks"):
        LarryMultimodalTrainer.prepare_model_config(
            template, _trainer_config(collator, tmp_path, ["voice_text", "text_voice", "text_only"])
        )


def test_the_voice_vocab_comes_from_the_speech_encoders_config(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"fsq_dims": 2, "fsq_levels": 5}))
    config = _trainer_config(collator, tmp_path, ["text_voice"], speech_encoder_model_path=str(tmp_path))

    built = LarryMultimodalTrainer.prepare_model_config(_template(collator, ["text_voice"]), config)

    assert (built.voice_prelude_config.vocab_size, built.voice_coda_config.vocab_size) == (25, 26)


def test_the_run_state_builds_the_model_from_the_trainers_derived_config(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    reset_accelerate_state()
    definition = TrainingDefinition(
        training=TrainingRunConfig(metrics_backend="tensorboard"),
        trainer=_trainer_config(collator, tmp_path, ["voice_text"]),
        model=_template(collator, ["voice_text"]),
    )

    state = TrainingRunState(definition, tmp_path / "run")

    model = state.trainer.accelerator.unwrap_model(state.trainer.model)
    assert type(state.trainer) is LarryMultimodalTrainer
    assert model.voice_coda is None and model.image_prelude is None


def _train_on_two_processes(processor_model_path: str, log_dir: str) -> None:
    from larry.multimodal.config.data.dataloading.dataloading_configs import PretrainingMultimodalCollatorConfig

    config = PretrainingMultimodalCollatorConfig(processor_model_path=processor_model_path)
    collator = PretrainingMultimodalCollator(config)
    # a budget of about one row per batch, so some ranks' batches have no voice
    trainer = _trainer(
        collator, Path(log_dir), ["text_voice", "text_only"], max_steps=4, max_batch_tokens=12, written=True
    )
    trainer.run()
    assert trainer.train_state.global_step == 4


def test_ranks_whose_batches_skip_a_modality_still_train_together(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    write_sources(tmp_path / "data", ["text_voice", "text_only"])

    spawn_launcher(_train_on_two_processes, args=(collator.config.processor_model_path, str(tmp_path)))
