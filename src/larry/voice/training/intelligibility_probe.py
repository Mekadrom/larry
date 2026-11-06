import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch

from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.common.training.viz_callbacks import pick_rows
from larry.voice.config.evaluation.evaluation_configs import IntelligibilityConfig
from larry.voice.evaluation.intelligibility import IntelligibilityScorer
from larry.voice.evaluation.speech_recognition import SpeechRecognizer

type ClipRenderer = Callable[[dict[str, Any]], Mapping[str, list[torch.Tensor]]]


class IntelligibilityProbe:
    def __init__(
            self,
            recognizers: Sequence[SpeechRecognizer],
            scorer: IntelligibilityScorer,
            dataset: IndexedDataset[Any],
            source_names: Sequence[str],
            collate_fn: Callable[[list[Any]], dict[str, Any]],
            config: IntelligibilityConfig,
    ) -> None:
        self.recognizers = list(recognizers)
        self.scorer = scorer
        self.dataset = dataset
        self.source_names = list(source_names)
        self.collate_fn = collate_fn
        self.config = config
        self.rows = pick_rows(dataset.offsets, config.clips_per_source)
        self.fixed_transcripts: dict[str, dict[str, list[str]]] = {}

    @classmethod
    def from_config(
            cls,
            config: IntelligibilityConfig,
            dataset: IndexedDataset[Any],
            source_names: Sequence[str],
            collate_fn: Callable[[list[Any]], dict[str, Any]],
            device: torch.device,
    ) -> IntelligibilityProbe | None:
        if not config.recognizers:
            return None

        recognizers = [SpeechRecognizer.from_config(recognizer, device) for recognizer in config.recognizers]
        return cls(recognizers, IntelligibilityScorer.whisper(), dataset, source_names, collate_fn, config)

    @functools.cached_property
    def references(self) -> tuple[list[str], list[str]]:
        references, groups = [], []
        for k in self.rows:
            source_idx, entry = self.dataset.entry(k)
            references.append(entry[self.config.transcript_column])
            groups.append(self.source_names[source_idx])

        return references, groups

    def batches(self) -> Iterator[dict[str, Any]]:
        for start in range(0, len(self.rows), self.config.batch_size):
            yield self.collate_fn([self.dataset[k] for k in self.rows[start:start + self.config.batch_size]])

    @torch.no_grad()
    def score(
            self,
            render: ClipRenderer,
            sample_rate: int,
            baseline: str,
            fixed: ClipRenderer | None = None,
    ) -> dict[str, float]:
        render_fixed = fixed is not None and not self.fixed_transcripts
        fixed_clips: dict[str, list[torch.Tensor]] = {}
        clips: dict[str, list[torch.Tensor]] = {}
        for batch in self.batches():
            if fixed is not None and render_fixed:
                for name, rendered in fixed(batch).items():
                    fixed_clips.setdefault(name, []).extend(rendered)

            for name, rendered in render(batch).items():
                clips.setdefault(name, []).extend(rendered)

        references, groups = self.references
        scores: dict[str, float] = {}
        for recognizer in self.recognizers:
            if render_fixed:
                self.fixed_transcripts[recognizer.name] = {
                    name: recognizer.transcribe(rendered, sample_rate) for name, rendered in fixed_clips.items()
                }

            transcripts = self.fixed_transcripts.get(recognizer.name, {}) | {
                name: recognizer.transcribe(rendered, sample_rate) for name, rendered in clips.items()
            }
            for metric, value in self.scorer.compare(references, transcripts, baseline, groups).items():
                scores[f"{recognizer.name}/{metric}"] = value

        return scores
