import dataclasses


@dataclasses.dataclass(kw_only=True)
class LarryProcessorConfig:
    text_tokenizer_model_file_path: str
    # both must match the model the processor feeds: one slot per patch, one slot per group of units
    image_patch_size: int = 2
    voice_units_per_position: int = 1
