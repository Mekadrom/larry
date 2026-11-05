from collections.abc import Callable, Sequence

from larry.voice.evaluation.transcripts import ErrorCounts, error_counts


class IntelligibilityScorer:
    """Scores transcripts of original and regenerated audio against references, overall and per group."""

    def __init__(self, normalize: Callable[[str], str]) -> None:
        self.normalize = normalize

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
            originals: Sequence[str],
            regenerated: Sequence[str],
            groups: Sequence[str],
    ) -> dict[str, float]:
        original = self.totals(self.counts(originals, references), groups)
        generated = self.totals(self.counts(regenerated, references), groups)
        scores: dict[str, float] = {}
        for key, base in original.items():
            for metric in ("wer", "cer"):
                before, after = getattr(base, metric), getattr(generated[key], metric)
                if before is None or after is None:
                    continue

                scores[f"{metric}_original{key}"] = before
                scores[f"{metric}_generated{key}"] = after
                scores[f"{metric}_gap{key}"] = after - before

        return scores
