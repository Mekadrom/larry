import dataclasses
import math
from typing import Literal

from larry.multimodal.config.model.model_configs import LarryMultimodalModelConfig
from larry.multimodal.processor.processor import EOV, LarryProcessorBase

type Task = Literal["text_only", "text_image", "text_voice", "image_text", "voice_text"]
type Mode = Literal["text", "image", "voice"]

TASK_MODES: dict[Task, set[Mode]] = {
    "text_only": {"text"},
    "text_image": {"text", "image"},
    "image_text": {"text", "image"},
    "text_voice": {"text", "voice"},
    "voice_text": {"text", "voice"},
}

# beyond the text prelude and coda, which every task needs
TASK_MODULES: dict[Task, set[str]] = {
    "text_only": set(),
    "text_image": {"image_coda_config"},
    "image_text": {"image_prelude_config"},
    # the voice prelude teacher-forces generated units and ties the voice head's embeddings
    "text_voice": {"voice_prelude_config", "voice_coda_config"},
    "voice_text": {"voice_prelude_config"},
}

MEDIA_MODULES = (
    "image_prelude_config", "image_coda_config", "voice_prelude_config", "voice_coda_config",
)


class LarryMultimodalModelConfigBuilder:
    def __init__(self, template: LarryMultimodalModelConfig) -> None:
        self.template = template

    def required_modules(self) -> set[str]:
        unknown = set(self.template.include_tasks) - set(TASK_MODES)
        if unknown:
            raise ValueError(f"unknown include_tasks: {sorted(unknown)}")

        used_modes = set().union(*(TASK_MODES[task] for task in self.template.include_tasks))
        missing_modes = used_modes - set(self.template.include_modes)
        if missing_modes:
            raise ValueError(f"include_tasks use modes missing from include_modes: {sorted(missing_modes)}")

        unused_modes = set(self.template.include_modes) - used_modes
        if unused_modes:
            raise ValueError(f"include_modes lists modes no task uses: {sorted(unused_modes)}")

        return set().union(*(TASK_MODULES[task] for task in self.template.include_tasks))

    def build(
            self,
            processor: LarryProcessorBase,
            fsq_levels: list[int] | None = None,
            latent_channels: int | None = None,
            unit_samples: int | None = None,
    ) -> LarryMultimodalModelConfig:
        required = self.required_modules()
        absent = sorted(name for name in required if getattr(self.template, name) is None)
        if absent:
            raise ValueError(f"include_tasks need module configs the template doesn't define: {absent}")

        # the text vocab is the tokenizer's plus the specials appended above it
        text_vocab_size = processor.text_tokenizer.vocab_size() + EOV + 1
        updates: dict[str, object | None] = {
            "text_prelude_config": dataclasses.replace(self.template.text_prelude_config, vocab_size=text_vocab_size),
            "text_coda_config": dataclasses.replace(self.template.text_coda_config, vocab_size=text_vocab_size),
        }
        for name in MEDIA_MODULES:
            module = getattr(self.template, name)
            if name in required:
                updates[name] = self.fit(name, module, processor, fsq_levels, latent_channels, unit_samples)
            else:
                updates[name] = None

        return dataclasses.replace(self.template, **updates)

    @staticmethod
    def fit(
            name: str,
            module: object,
            processor: LarryProcessorBase,
            fsq_levels: list[int] | None,
            latent_channels: int | None,
            unit_samples: int | None = None,
    ) -> object:
        # every size here is decided by the processor or an upstream encoder, not by hand in the template
        if name.startswith("image"):
            updates = {"patch_size": processor.config.image_patch_size}
            if latent_channels is not None:
                updates["latent_channels"] = latent_channels

            return dataclasses.replace(module, **updates)  # pyright: ignore[reportArgumentType]

        updates = {"units_per_position": processor.config.voice_units_per_position}
        if fsq_levels is not None:
            n_units = math.prod(fsq_levels)
            if name == "voice_prelude_config":
                updates |= {"vocab_size": n_units, "fsq_levels": list(fsq_levels)}
            else:
                updates["vocab_size"] = n_units + 1

        if name == "voice_coda_config" and unit_samples is not None:
            hop_length = module.prosody_hop_length  # pyright: ignore[reportAttributeAccessIssue]
            if unit_samples % hop_length:
                raise ValueError(f"a unit spans {unit_samples} samples, not a whole number of {hop_length}-sample frames")

            updates["prosody_frames_per_unit"] = unit_samples // hop_length

        return dataclasses.replace(module, **updates)  # pyright: ignore[reportArgumentType]
