from typing import Any

import torch
from datasets import List, Value
from datasets.features.features import FeatureType
from transformers import AutoTokenizer

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.mappers import UrlMapper, SingleColumnMapper
from larry.text.config.preprocessing.text_mapper_configs import TextTokenizingMapperConfig, \
    UrlTextMapperConfig


class UrlTextMapper(UrlMapper[str, UrlTextMapperConfig], SingleColumnMapper[str, str | None, UrlTextMapperConfig]):
    def user_agent(self) -> str:
        return "larry-text-downloader/1.0 (dataset research; contact via github.com/Mekadrom)"

    def validate_content(self, example: str, content: bytes) -> None:
        pass

    def extract_content(self, example: str, content: bytes) -> str:
        try:
            return content.decode("utf-8", errors="replace")
        except (LookupError, TypeError):
            self.log.error(f"content encoding error, content was not utf-8 for content of size {len(content)}")
            return content.decode(errors="replace")

    def output_feature(self) -> FeatureType | None:
        return Value("string")


# noinspection PyTypeChecker
class TextTokenizingMapper(SingleColumnMapper[list[str], torch.Tensor | None, TextTokenizingMapperConfig]):
    def __init__(self, provenance: Provenance, config: TextTokenizingMapperConfig) -> None:
        super().__init__(provenance, config)
        # noinspection PyNoneFunctionAssignment
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.log.info(f"Using tokenizer with vocab_size={len(self.tokenizer)}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        encoded = self.tokenizer(batch[self.config.input_column], add_special_tokens=False)
        return {self.config.output_column: encoded["input_ids"]}

    def output_feature(self) -> FeatureType | None:
        return List(Value("uint16"))
