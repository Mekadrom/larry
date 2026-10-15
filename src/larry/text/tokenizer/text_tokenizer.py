from abc import abstractmethod, ABC
from typing import ClassVar

import sentencepiece

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class TextTokenizer(Registrable, ABC, root=True):
    """Superclass for wrapping text tokenizers trained within this project."""
    REGISTRY: ClassVar[TypeRegistry[TextTokenizer]]

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Wraps the underlying tokenizer's encode functionality to turn text into token ids."""
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Wraps the underlying tokenizer's encode functionality to turn token ids into text."""
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the number of unique token ids that this tokenizer can produce or accept."""
        ...

    @abstractmethod
    def id_to_piece(self, token_id: int) -> str:
        """Converts a single integer token id into its corresponding token string."""
        ...


class SentencePieceTokenizer(TextTokenizer):
    """Wraps a ``SentencePieceProcessor`` tokenizer with the given model_file."""

    def __init__(self, model_file: str) -> None:
        """Loads the delegate tokenizer."""
        self.sp = sentencepiece.SentencePieceProcessor(model_file=model_file)

    def encode(self, text: str) -> list[int]:
        """Delegates to the encode function."""
        return self.sp.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Delegates to the decode function"""
        return self.sp.decode(token_ids)

    def vocab_size(self) -> int:
        """Delegates to the get_piece_size function."""
        return self.sp.get_piece_size()

    def id_to_piece(self, token_id: int) -> str:
        """Delegates to the id_to_piece function."""
        return self.sp.id_to_piece(token_id)
