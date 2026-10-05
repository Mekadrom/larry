import re
from typing import Any

import torch
from datasets import List, Value
from datasets.features.features import FeatureType
from transformers import AutoTokenizer

from larry.common.data.preprocessing.mappers import Mapper
from larry.text.config.preprocessing.text_mapper_configs import TextTokenizingMapperConfig, \
    TextMapperConfig


class TextMapper[I, O, C: TextMapperConfig = TextMapperConfig](Mapper[I, O, C]):
    ...


class TextNormalizingMapper(TextMapper[str | None, str | None, TextMapperConfig]):
    def preprocess_example(self, example: str | None) -> str | None:
        if not example:
            return None
        text = example.strip()
        if not text:
            return None

        text = self._remove_symbols(text)
        text = self._collapse_whitespace(text)
        text = self._normalize_casing(text)
        text = self._capitalize_start(text)
        text = self._punctuate_end(text)

        return text

    @staticmethod
    def _remove_symbols(text: str) -> str:
        # remove brackets, carets, slashes, pipes, asterisks, underscores, and tilde/grave. may leave whitespace idk
        return re.sub(r'[*_\[\]{}<>|\\~`^]', '', text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        # collapse multiple-whitespace runs to one space each, and trim again to remove any errant newline->space
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _normalize_casing(text: str) -> str:
        # only normalize casing if the text appears to be mostly caps or all caps (>80% uppercase letters)
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.8:
            text = text.lower()
        return text

    @staticmethod
    def _capitalize_start(text: str) -> str:
        # replace the first alphabetic character with its uppercase counterpart; works inside quotes
        return re.sub(
            r'(^|[.!?]\s+)(["\']*)([a-z])',
            lambda m: m.group(1) + m.group(2) + m.group(3).upper(),
            text
        )

    @staticmethod
    def _punctuate_end(text: str) -> str:
        if text and text[-1].isalnum():
            text = text + '.'
        return text


# noinspection PyTypeChecker
class TextTokenizingMapper(TextMapper[list[str], torch.Tensor | None, TextTokenizingMapperConfig]):
    def __init__(self, config: TextTokenizingMapperConfig) -> None:
        super().__init__(config)
        # noinspection PyNoneFunctionAssignment
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.log.info(f"Using tokenizer with vocab_size={len(self.tokenizer)}")

    def output_feature(self) -> FeatureType | None:
        return List(Value("uint16"))

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        encoded = self.tokenizer(batch[self.config.input_column], add_special_tokens=False)
        return {self.config.output_column: encoded["input_ids"]}
