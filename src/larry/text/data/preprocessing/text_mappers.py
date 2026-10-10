import functools
from typing import Any

import torch
from datasets import List, Value
from datasets.features.features import FeatureType
from transformers import AutoTokenizer

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig
from larry.common.data.preprocessing.mappers import SingleColumnMapper, UrlMapper
from larry.text.config.preprocessing.text_mapper_configs import TextTokenizingMapperConfig, \
    UrlTextMapperConfig, CTCTextNormalizingMapperConfig
from larry.voice.data.preprocessing.normalization import TextNormalizer


class TextMapper[I, O, C: SingleColumnMapperConfig = SingleColumnMapperConfig](SingleColumnMapper[I, O, C]):
    ...


class UrlTextMapper(UrlMapper[str, UrlTextMapperConfig], TextMapper[str, str | None, UrlTextMapperConfig]):
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


# noinspection PyTypeChecker
class TextTokenizingMapper(TextMapper[list[str], torch.Tensor | None, TextTokenizingMapperConfig]):
    def __init__(self, provenance_columns: list[str], config: TextTokenizingMapperConfig) -> None:
        super().__init__(provenance_columns, config)
        # noinspection PyNoneFunctionAssignment
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.log.info(f"Using tokenizer with vocab_size={len(self.tokenizer)}")

    def output_feature(self) -> FeatureType | None:
        return List(Value("uint16"))

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        encoded = self.tokenizer(batch[self.config.input_column], add_special_tokens=False)
        return {self.config.output_column: encoded["input_ids"]}
