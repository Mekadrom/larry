from typing import Any

from datasets import Audio, Dataset

from larry.common.config.data.preprocessing.pipeline_config import Provenance, ProvenanceAncestry
from larry.voice.config.preprocessing.voice_mapper_configs import SpeechSegmentationAligningMapperConfig
from larry.voice.data.preprocessing.normalization import TextNormalizer
from larry.voice.data.preprocessing.transcript import Transcript, Turn
from larry.voice.data.preprocessing.voice_mappers import SpeechSegmentationAligningMapper


class TwoSegmentsPerRowMapper(SpeechSegmentationAligningMapper[SpeechSegmentationAligningMapperConfig]):
    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {k: [] for k in self.config.copy_columns + self.output_columns}
        for i, text in enumerate(batch[self.config.text_column]):
            for k in range(2):
                out["audio"].append({"bytes": b"x", "path": None})
                out["text_original"].append(f"{text}-{k}")
                out["text_normalized"].append(f"{text}-{k}".upper())
                out["duration_s"].append(1.0)
                out["start_s"].append(float(k))
                out["end_s"].append(float(k + 1))
                out["speaker_id"].append("s")
                out["avg_word_score"].append(-0.5)
                for c in self.config.copy_columns:
                    out[c].append(batch[c][i])
        return out

    def make_normalizer(self) -> TextNormalizer:
        return TextNormalizer()

    def new_transcript(self, turns: list[Turn], batch: dict[str, list[Any]], index: int) -> Transcript:
        raise NotImplementedError


def test_segmentation_mapper_replaces_rows_and_keeps_only_copied_columns() -> None:
    dataset = Dataset.from_dict({
        "audio": [{"bytes": b"a", "path": None}, {"bytes": b"b", "path": None}],
        "text_original": ["one", "two"],
        "docket": ["d1", "d2"],
        "pdf_bytes": [b"%PDF-1", b"%PDF-2"],
    }).cast_column("audio", Audio(decode=False))
    config = SpeechSegmentationAligningMapperConfig(
        num_proc=1, batch_size=1, writer_batch_size=1, cache_results=False, copy_columns=["docket"],
    )
    mapper = TwoSegmentsPerRowMapper(
        Provenance(
            id_columns=[],
            produced_on="",
            git_commit="",
            git_is_dirty=None,
            ancestry=ProvenanceAncestry(datasets=[], models=[])
        ),
        config,
    )

    out = mapper.preprocess_dataset(dataset)

    assert len(out) == 4
    assert sorted(out.column_names) == sorted(["docket", *mapper.output_columns])
    assert isinstance(out, Dataset)
    assert out.features["audio"] == Audio(decode=False)
    assert out["docket"] == ["d1", "d1", "d2", "d2"]
    assert out["text_original"] == ["one-0", "one-1", "two-0", "two-1"]
