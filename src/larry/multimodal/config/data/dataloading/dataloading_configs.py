import dataclasses


@dataclasses.dataclass(kw_only=True)
class PretrainingMultimodalCollatorConfig:
    processor_model_path: str


@dataclasses.dataclass(kw_only=True)
class InferenceMultimodalCollatorConfig:
    processor_model_path: str
