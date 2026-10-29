from typing import Any

import torch
from datasets import Dataset, DatasetDict, Features, List, Value
from datasets.features.features import FeatureType
from transformers import AutoTokenizer

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.mappers import Mapper, SingleColumnMapper
from larry.common.data.preprocessing.url_mappers import UrlMapper
from larry.common.data.utils import bot_user_agent
from larry.text.config.preprocessing.text_mapper_configs import (
    CaptionSelectingMapperConfig,
    TextTokenizingMapperConfig,
    UrlTextMapperConfig,
    VisualDescriptionExtractingMapperConfig,
)
from larry.text.data.preprocessing.visual_text import VisualTextExtractor


class UrlTextMapper(UrlMapper[str, UrlTextMapperConfig]):
    """Implementation of ``UrlTextMapper`` that processes native text content."""

    def user_agent(self) -> str:
        """Overrides the user agent to point to pipelines using this mapper in this project."""
        return bot_user_agent("larry-text-downloader")

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


class VisualDescriptionExtractingMapper(Mapper[VisualDescriptionExtractingMapperConfig]):
    def __init__(self, provenance: Provenance, config: VisualDescriptionExtractingMapperConfig) -> None:
        super().__init__(provenance, config)
        self.extractor = VisualTextExtractor(config)

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        extracted = [self.extractor.extract(text) for text in batch[self.config.input_column]]
        return {
            self.config.output_column: [text for text, _ in extracted],
            self.config.retained_output_column: [retained for _, retained in extracted],
        }

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        return Features({
            self.config.output_column: Value("string"),
            self.config.retained_output_column: Value("float32"),
        })


class CaptionSelectingMapper(Mapper[CaptionSelectingMapperConfig]):
    def candidate(self, value: Any) -> str | None:
        if isinstance(value, list):
            value = next((v for v in value if isinstance(v, str) and v.strip()), None)

        if not isinstance(value, str):
            return None

        value = value.strip()
        return value if len(value) >= self.config.min_characters else None

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        captions: list[str | None] = []
        sources: list[str | None] = []
        for i in range(len(batch[self.config.caption_columns[0]])):
            chosen = next(
                ((c, text) for c in self.config.caption_columns if (text := self.candidate(batch[c][i])) is not None),
                (None, None),
            )
            sources.append(chosen[0])
            captions.append(chosen[1])

        return {self.config.output_column: captions, self.config.source_output_column: sources}

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        return Features({
            self.config.output_column: Value("string"),
            self.config.source_output_column: Value("string"),
        })
