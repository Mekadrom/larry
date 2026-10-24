import dataclasses


@dataclasses.dataclass(kw_only=True)
class PretrainingMultimodalCollatorConfig:
    processor_model_path: str
    image_patch_size: tuple[int, int] = (2, 2)