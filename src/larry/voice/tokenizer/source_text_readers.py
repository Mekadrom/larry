import pyarrow.compute as pc
from datasets import Dataset as HFDataset

from larry.common.data.dataloading.indexed_dataset import indexed_column
from larry.common.training.tokenizer_training import SourceTextReader
from larry.voice.config.data.voice_data_configs import ASRSourceConfig


class ASRSourceTextReader(SourceTextReader[ASRSourceConfig]):
    unit = "seconds"

    def columns(self, source: ASRSourceConfig) -> list[str]:
        return [*super().columns(source), source.duration_column]

    def size(self, source: ASRSourceConfig, index: HFDataset, data: HFDataset | None, texts: list[str]) -> float:
        return float(pc.sum(indexed_column(index, data, source.duration_column)).as_py() or 0.0)
