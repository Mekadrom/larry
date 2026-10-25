import dataclasses
from pathlib import Path

from larry.common.config.config import Config
from larry.common.config.model.model_configs import ModelConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig, TokenizerTrainingConfig
from larry.multimodal.config.criterion.criterion_configs import LarryMultimodalCriterionConfig
from larry.multimodal.config.data.data_configs import MultimodalMixtureConfig


@dataclasses.dataclass(kw_only=True)
class MultimodalVizConfig(Config):
    examples_per_source: int = 2
    max_new_tokens: int = 48
    voice_temperature: float = 0.8
    # generated units run this many times the reference's groups, plus slack, before being cut off
    voice_length_budget: float = 1.5
    image_steps: int = 16
    image_cfg_scale: float = 2.0
    seed: int = 0
    # speech decoder and vocoder checkpoints turn units into audio; without them voice is judged on units alone
    speech_decoder_model_path: str | None = None
    vocoder_model_path: str | None = None
    vocoder_steps: int = 16


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalTrainerConfig(LarryTrainerConfig[MultimodalMixtureConfig], TokenizerTrainingConfig):
    processor_model_path: str
    # used by tokenizer-training to build the processor; training then checks the saved processor still matches
    text_vocab_size: int = 16384
    image_patch_size: int = 2
    voice_units_per_position: int = 1
    # the encoder whose FSQ produced the datasets' unit ids; its config sets the voice vocab
    speech_encoder_model_path: str | None = None
    # channels of the VAE latents the datasets store
    image_latent_channels: int | None = None
    criterion: LarryMultimodalCriterionConfig = dataclasses.field(default_factory=LarryMultimodalCriterionConfig)
    # each checkpoint also gets a from_pretrained-ready copy of the model with its processor
    export_on_save: bool = True
    viz: MultimodalVizConfig = dataclasses.field(default_factory=MultimodalVizConfig)

    def tokenizer_vocab_size(self, model_config: ModelConfig) -> int:
        # the model's text vocab is this plus the specials, and is set from the processor at training time
        return self.text_vocab_size

    def tokenizer_output_prefix(self) -> Path | None:
        return Path(self.processor_model_path) / "sentencepiece" / "text"

    def write_tokenizer_artifacts(self, model_file: Path) -> None:
        # the processor holds the tokenizer with the stretch settings, so model and data read one definition
        from larry.multimodal.config.processor_configs import LarryProcessorConfig
        from larry.multimodal.processor.processor import LarryMultimodalPretrainProcessor

        processor = LarryMultimodalPretrainProcessor(LarryProcessorConfig(
            text_tokenizer_model_file_path=str(model_file),
            image_patch_size=self.image_patch_size,
            voice_units_per_position=self.voice_units_per_position,
        ))
        processor.save_pretrained(self.processor_model_path)
