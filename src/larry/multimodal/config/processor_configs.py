import dataclasses


@dataclasses.dataclass(kw_only=True)
class LarryProcessorConfig:
    text_tokenizer_model_file_path: str
