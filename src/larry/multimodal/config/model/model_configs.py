import dataclasses
from typing import Literal

from larry.common.config.model.model_configs import LarryRecurrentTrunkConfig, ModelConfig, VocabularyConfig
from larry.voice.model.speech_decoder import LLMBlockModuleConfig


@dataclasses.dataclass(kw_only=True)
class TextModuleConfig(VocabularyConfig, LLMBlockModuleConfig):
    n_layers: int = 2


@dataclasses.dataclass(kw_only=True)
class ImageModuleConfig(LLMBlockModuleConfig):
    n_layers: int = 2
    patch_size: int = 2


@dataclasses.dataclass(kw_only=True)
class VoiceModuleConfig(VocabularyConfig, LLMBlockModuleConfig):
    n_layers: int = 2


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalModelConfig(ModelConfig):
    text_prelude_config: TextModuleConfig
    image_prelude_config: ImageModuleConfig | None = None
    voice_prelude_config: VoiceModuleConfig | None = None
    recurrent_trunk_config: LarryRecurrentTrunkConfig
    text_coda_config: TextModuleConfig
    image_coda_config: ImageModuleConfig | None = None
    voice_coda_config: VoiceModuleConfig | None = None

    include_modes: list[Literal["text", "image", "video"]]
    include_tasks: list[Literal["text_only", "text_image", "text_voice", "image_text", "voice_text"]]
