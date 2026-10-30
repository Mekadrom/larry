import dataclasses
import math
from typing import Literal

from larry.common.config.model.model_configs import (
    FlowMatchingConfig,
    LarryConditionalEncoderBlockModuleConfig,
    LarryRecurrentTrunkConfig,
    LLMBlockModuleConfig,
    ModelConfig,
    VocabularyConfig,
)


@dataclasses.dataclass(kw_only=True)
class TextModuleConfig(VocabularyConfig, LLMBlockModuleConfig):
    n_layers: int = 2


@dataclasses.dataclass(kw_only=True)
class ImagePreludeConfig(LLMBlockModuleConfig):
    n_layers: int = 2
    patch_size: int = 2
    latent_channels: int = 16
    is_causal: bool = False


@dataclasses.dataclass(kw_only=True)
class ImageCodaConfig(LarryConditionalEncoderBlockModuleConfig, FlowMatchingConfig):
    n_layers: int = 2
    patch_size: int = 2
    latent_channels: int = 16
    max_query_grid: int = 64
    cfg_dropout_rate: float = 0.1


@dataclasses.dataclass(kw_only=True)
class VoiceModuleConfig(VocabularyConfig, LLMBlockModuleConfig):
    n_layers: int = 2
    units_per_position: int = 1


@dataclasses.dataclass(kw_only=True)
class VoicePreludeConfig(VoiceModuleConfig):
    fsq_levels: list[int] = dataclasses.field(default_factory=lambda: [3] * 8)
    conv_kernel_size: int = 4
    group_combine: Literal["concat", "sum"] = "concat"


@dataclasses.dataclass(kw_only=True)
class VoiceCodaConfig(VoiceModuleConfig):
    depth_n_layers: int = 2
    prosody_head: bool = True
    prosody_hop_length: int = 320
    prosody_frames_per_unit: int = 4
    prosody_hidden_dim: int = 1024
    n_aperiodicity_bands: int = 8


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalModelConfig(ModelConfig):
    text_prelude_config: TextModuleConfig
    image_prelude_config: ImagePreludeConfig | None = None
    voice_prelude_config: VoicePreludeConfig | None = None
    recurrent_trunk_config: LarryRecurrentTrunkConfig
    text_coda_config: TextModuleConfig
    image_coda_config: ImageCodaConfig | None = None
    voice_coda_config: VoiceCodaConfig | None = None

    include_modes: list[Literal["text", "image", "voice"]]
    include_tasks: list[Literal["text_only", "text_image", "text_voice", "image_text", "voice_text"]]

    def __post_init__(self) -> None:
        if self.text_coda_config.vocab_size != self.text_prelude_config.vocab_size:
            raise ValueError("the text coda's lm_head is tied to the text prelude's wte; their vocab sizes must match")

        modules = [
            self.text_prelude_config,
            self.image_prelude_config,
            self.voice_prelude_config,
            self.text_coda_config,
            self.image_coda_config,
            self.voice_coda_config,
        ]
        if any(m is not None and m.d_model != self.recurrent_trunk_config.d_model for m in modules):
            raise ValueError("every prelude and coda must share the recurrent trunk's d_model")

        voice_prelude = self.voice_prelude_config
        if voice_prelude is not None and voice_prelude.vocab_size != math.prod(voice_prelude.fsq_levels):
            raise ValueError("the voice prelude's vocab must be every FSQ code: the product of fsq_levels")

        if self.voice_coda_config is not None:
            if voice_prelude is None:
                raise ValueError("the voice coda's head is tied to the voice prelude's unit embeddings; it needs one")

            if self.voice_coda_config.vocab_size != voice_prelude.vocab_size + 1:
                raise ValueError("the voice coda's vocab is the voice prelude's units plus one class for <eov>")

            if self.voice_coda_config.units_per_position != voice_prelude.units_per_position:
                raise ValueError("the voice prelude and coda must group the same units_per_position")

        if self.image_prelude_config is not None and self.image_coda_config is not None:
            prelude_layout = (self.image_prelude_config.patch_size, self.image_prelude_config.latent_channels)
            if prelude_layout != (self.image_coda_config.patch_size, self.image_coda_config.latent_channels):
                raise ValueError("the image prelude and coda must agree on patch_size and latent_channels")

    @property
    def voice_units_per_position(self) -> int:
        voice_config = self.voice_prelude_config or self.voice_coda_config
        return voice_config.units_per_position if voice_config is not None else 1

    @property
    def image_patch_size(self) -> int | None:
        image_config = self.image_prelude_config or self.image_coda_config
        return image_config.patch_size if image_config is not None else None
