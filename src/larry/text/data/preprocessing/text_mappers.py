from typing import Any

import torch
from datasets import List, Value
from datasets.features.features import FeatureType
from transformers import AutoTokenizer

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.mappers import SingleColumnMapper
from larry.common.data.preprocessing.url_mappers import UrlMapper
from larry.text.config.preprocessing.text_mapper_configs import TextTokenizingMapperConfig, \
    UrlTextMapperConfig


class UrlTextMapper(UrlMapper[str, UrlTextMapperConfig], SingleColumnMapper[str, str | None, UrlTextMapperConfig]):
    """Implementation of ``UrlTextMapper`` that processes native text content."""

    def user_agent(self) -> str:
        """Overrides the user agent to point to pipelines using this mapper in this project."""
        return "larry-text-downloader/1.0 (dataset research; contact via github.com/Mekadrom)"

    def validate_content(self, example: str, content: bytes) -> None:
        """No validation can be done or is required in addition to the superclass'."""
        pass

    def extract_content(self, example: str, content: bytes) -> str:
        """Simply reads the bytes as utf-8 text. Errors related to utf-8 decoding are retried once."""
        try:
            return content.decode("utf-8", errors="replace")
        except (LookupError, TypeError):
            self.log.error(f"content encoding error, content was not utf-8 for content of size {len(content)}")
            return content.decode(errors="replace")

    def output_feature(self) -> FeatureType | None:
        """Just in case, forces the output feature type for this mapper's output column to string."""
        return Value("string")


# noinspection PyTypeChecker
class TextTokenizingMapper(SingleColumnMapper[list[str], torch.Tensor | None, TextTokenizingMapperConfig]):
    """A mapper that maps native text to an array of integers."""

    def __init__(self, provenance: Provenance, config: TextTokenizingMapperConfig) -> None:
        """Loads the tokenizer."""
        super().__init__(provenance, config)
        # noinspection PyNoneFunctionAssignment
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.log.info(f"Using tokenizer with vocab_size={len(self.tokenizer)}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        """Preprocesses an entire batch using the delegate tokenizer's batch processing."""
        encoded = self.tokenizer(batch[self.config.input_column], add_special_tokens=False)
        return {self.config.output_column: encoded["input_ids"]}

    def output_feature(self) -> FeatureType | None:
        """Makes the output feature type for this mapper's output column a list of unsigned 32-bit integers."""
        return List(Value("uint32"))
