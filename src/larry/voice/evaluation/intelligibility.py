from collections.abc import Callable, Mapping, Sequence

from larry.voice.evaluation.transcripts import (
    ErrorCounts,
    error_counts,
    load_whisper_normalizer,
    normalize_for_scoring,
)


class IntelligibilityScorer:
    def __init__(self, normalize: Callable[[str], str]) -> None:
        self.normalize = normalize

    @classmethod
    def whisper(cls) -> IntelligibilityScorer:
        normalizer = load_whisper_normalizer()
        return cls(lambda text: normalize_for_scoring(normalizer, text))

    def counts(self, hypotheses: Sequence[str], references: Sequence[str]) -> list[ErrorCounts]:
        return [
            error_counts(self.normalize(hypothesis), self.normalize(reference))
            for hypothesis, reference in zip(hypotheses, references, strict=True)
        ]

    @staticmethod
    def totals(counts: Sequence[ErrorCounts], groups: Sequence[str]) -> dict[str, ErrorCounts]:
        totals: dict[str, ErrorCounts] = {}
        for count, group in zip(counts, groups, strict=True):
            for key in ("", f"/{group}"):
                totals[key] = totals[key] + count if key in totals else count

        return totals

    def compare(
            self,
            references: Sequence[str],
            transcripts: Mapping[str, Sequence[str]],
            baseline: str,
            groups: Sequence[str],
    ) -> dict[str, float]:
        totals = {name: self.totals(self.counts(heard, references), groups) for name, heard in transcripts.items()}
        scores: dict[str, float] = {}
        for name, by_group in totals.items():
            for key, count in by_group.items():
                for metric in ("wer", "cer"):
                    value, base = getattr(count, metric), getattr(totals[baseline][key], metric)
                    if value is None:
                        continue

                    scores[f"{metric}_{name}{key}"] = value
                    if name != baseline and base is not None:
                        scores[f"{metric}_gap_{name}{key}"] = value - base

        return scores
