from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from datasets.features.features import Value, Features

from larry.common.data.preprocessing.mappers import SingleColumnMapper
from larry.common.data.utils import features_of
from larry.text.config.preprocessing.text_mapper_configs import SCOTUSTranscriptPdfTextMapperConfig
from larry.text.data.preprocessing.scotus import transcript


class SCOTUSTranscriptPdfTextMapper(SingleColumnMapper[bytes, str, SCOTUSTranscriptPdfTextMapperConfig]):
    def __init__(self, provenance_columns: list[str], config: SCOTUSTranscriptPdfTextMapperConfig) -> None:
        super().__init__(provenance_columns, config)
        self.download_cache_dir = Path(self.config.download_cache_dir).expanduser().resolve()

    def validate(self) -> None:
        if not self.config.download_cache_dir:
            raise ValueError(f"download_cache_dir must be specified for {self.__class__.__name__}")
        if not self.config.input_column:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")
        if not self.config.docket_column:
            raise ValueError(f"docket_column must be specified for {self.__class__.__name__}")
        if not self.config.output_column:
            raise ValueError(f"output_column must be specified for {self.__class__.__name__}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        inputs = batch[self.config.input_column]
        dockets = batch[self.config.docket_column]
        transcripts = []
        for url_hash, docket in zip(inputs, dockets):
            tr = transcript.parse(str(self.download_cache_dir / url_hash))
            transcripts.append("\n".join([f"{turn.speaker}: {turn.text}" for turn in tr.turns]))
        return {
            self.config.output_column: transcripts,
        }

    def preprocessing_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        features = super().preprocessing_features(dataset)
        if features is None:
            features = features_of(dataset).copy()
        for column in self.remove_columns:
            features.pop(column, None)
        features.update({
            self.config.output_column: Value("string"),
        })
        return features
