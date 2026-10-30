import dataclasses

from larry.common.config.criteria.criterion_configs import CriterionConfig


@dataclasses.dataclass(kw_only=True)
class CTCCriterionConfig(CriterionConfig):
    """Config for CTC criteria."""

    blank: int
    zero_infinity: bool = True

    ctc_layer_loss_weights: dict[str, float] = dataclasses.field(default_factory=lambda: {
        "encoder2_norm": 1.0,
    })


@dataclasses.dataclass(kw_only=True)
class SpeechDecoderCriterionConfig(CriterionConfig):
    kl_weight: float = 1e-4
    kl_warmup_steps: int = 10_000
    prior_weight: float = 1.0
    f0_range_weight: float = 1.0


@dataclasses.dataclass(kw_only=True)
class VocoderCriterionConfig(CriterionConfig):
    flow_weight: float = 1.0
    stft_weight: float = 0.0
    mel_weight: float = 0.0
    stft_fft_sizes: list[int] = dataclasses.field(default_factory=lambda: [512, 1024, 2048])
    estimate_t_power: float = 1.0
