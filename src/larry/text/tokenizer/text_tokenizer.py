from abc import abstractmethod, ABC
from typing import ClassVar

import sentencepiece

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class TextTokenizer(Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[TextTokenizer]]

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        ...


class SentencePieceTokenizer(TextTokenizer):
    def __init__(self, model_file: str) -> None:
        self.sp = sentencepiece.SentencePieceProcessor(model_file=model_file)

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.sp.decode(token_ids)

    def vocab_size(self) -> int:
        return self.sp.get_piece_size()