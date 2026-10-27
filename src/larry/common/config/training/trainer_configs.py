import dataclasses
from abc import ABC
from pathlib import Path
from typing import Any, Literal

from larry.common.config.config import Config, TaggedConfig
from larry.common.config.data.data_configs import DatasetMixtureConfig
from larry.common.config.model.model_configs import ModelConfig, VocabularyConfig


@dataclasses.dataclass(kw_only=True)
class LarryTrainerConfig[M: DatasetMixtureConfig[Any] = DatasetMixtureConfig](TaggedConfig, ABC):
    """Superclass config for all ``TrainerBase`` and ``LarryTrainer`` implementations."""

    mixture_config: M

    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16"
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    ddp_comm_hook: Literal["no", "fp16", "bf16", "power_sgd", "batched_power_sgd"] = "no"
    ddp_comm_state_option: dict[str, Any] = dataclasses.field(default_factory=dict)

    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_dataloader_workers: int = 8

    max_steps: int
    seed: int = 42

    logging_steps: int = 100
    eval_steps: int | None = None
    viz_steps: int | None = None
    save_steps: int = 1000

    optimizer_name: Literal["muon", "adamw"] = "muon"
    optimizer_config: dict[str, Any] = dataclasses.field(default_factory=dict)

    lr: float = 1e-4
    min_lr: float = 0
    weight_decay: float = 0.01

    adamw_betas: tuple[float, float] = (0.9, 0.999)

    muon_first_layer_names: list[str] = dataclasses.field(default_factory=list)
    muon_last_layer_names: list[str] = dataclasses.field(default_factory=list)
    muon_qk_clip_tau: float = 60.0
    muon_qk_clip_probe_every: int = 50
    muon_qk_clip_alpha: float = 0.5

    # seeds the flow-matching t and noise at eval, so flow losses compare across evals and across runs
    eval_noise_seed: int = 0

    warmup_steps: int = 0
    lr_scheduler_name: Literal["constant", "cosine"] = "constant"
    decay_start_step: int | None = None


class TokenizerTrainingConfig:
    """A trainer config whose text tokenizer is trained by ``tokenizer-training`` on its own train mixture."""

    def tokenizer_vocab_size(self, model_config: ModelConfig) -> int:
        if not isinstance(model_config, VocabularyConfig):
            raise TypeError(f"{type(model_config).__name__} has no text vocabulary to train a tokenizer for")

        return model_config.vocab_size

    def tokenizer_output_prefix(self) -> Path | None:
        """Where ``<prefix>.model`` is written when ``--output_prefix`` isn't given; None means it must be."""
        return None

    def write_tokenizer_artifacts(self, model_file: Path) -> None:
        """Anything the trainer reads besides the SentencePiece model itself; nothing by default."""
        ...


@dataclasses.dataclass(kw_only=True)
class TextTokenizerConfig(Config, TokenizerTrainingConfig):
    tokenizer_model_path: str

    def tokenizer_output_prefix(self) -> Path | None:
        return Path(self.tokenizer_model_path).with_suffix("")
