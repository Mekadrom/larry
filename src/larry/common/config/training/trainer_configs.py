import dataclasses
from abc import ABC
from typing import Literal, Any, ClassVar

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class LarryTrainerConfig(Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[LarryTrainerConfig]]

    dataset_config: dict[str, Any] = dataclasses.field(default_factory=dict)

    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0

    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_dataloader_workers: int = 8

    num_epochs: int = 1

    logging_steps: int = 100
    eval_steps: int | None = None
    viz_steps: int | None = None
    save_steps: int = 1000

    optimizer_name: Literal["muon", "adamw"] = "muon"
    optimizer_config: dict[str, Any] = dataclasses.field(default_factory=dict)

    lr: float = 1e-4
    min_lr: float = 0
    weight_decay: float = 0.01

    adamw_betas: tuple[float, float] = (0.99, 0.999)

    muon_first_layer_names: list[str] = dataclasses.field(default_factory=lambda: [
        "embed.weight"
    ])
    muon_last_layer_names: list[str] = dataclasses.field(default_factory=list)
    muon_qk_clip_tau: float = 60.0
    muon_qk_clip_probe_every: int = 50
    muon_qk_clip_alpha: float = 0.5

    warmup_steps: int = 0
    lr_scheduler_name: Literal["constant", "cosine"] = "constant"
