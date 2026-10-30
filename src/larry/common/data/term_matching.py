import re
import unicodedata
from collections.abc import Iterable


class TermMatcher:
    def __init__(self, terms: Iterable[str]) -> None:
        normalized = sorted({self.normalize(t) for t in terms if t.strip()}, key=len, reverse=True)
        alternatives = "|".join(re.escape(t) for t in normalized)
        self.pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)") if normalized else None

    @staticmethod
    def normalize(text: str) -> str:
        return unicodedata.normalize("NFC", text).casefold()

    def matches(self, texts: Iterable[str | None]) -> bool:
        if self.pattern is None:
            return False

        return any(self.pattern.search(self.normalize(t)) for t in texts if t)
