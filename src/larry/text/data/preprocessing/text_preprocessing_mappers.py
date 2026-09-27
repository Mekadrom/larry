import logging
import re
from abc import ABC

import torch
from transformers import AutoTokenizer

from larry.common.data.preprocessing import PreprocessingMapper
from larry.text.config.text_preprocessing_mapper_configs import TextTokenizationMapperConfig, \
    TextPreprocessingMapperConfig

log = logging.getLogger(__name__)


class TextPreprocessingMapper[I = None, O = None](PreprocessingMapper[I, O], ABC):
    config: TextPreprocessingMapperConfig

    def __init__(self, config: TextPreprocessingMapperConfig) -> None:
        super().__init__(config)


class TextNormalizingPreprocessingMapper(TextPreprocessingMapper[str, str]):
    def preprocess_example(self, example: str) -> str:
        text = example.strip()
        if not text:
            return text

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
class TextTokenizingPreprocessingMapper(TextPreprocessingMapper[list[str], dict[str, torch.Tensor]]):
    config: TextTokenizationMapperConfig

    def __init__(self, config: TextTokenizationMapperConfig) -> None:
        super().__init__(config)
        # noinspection PyNoneFunctionAssignment
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.log.info(f"Using tokenizer with vocab_size={len(self.tokenizer)}")

    def preprocess_example(self, example: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(example)
