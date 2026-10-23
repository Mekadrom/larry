from datasets import Dataset as HFDataset

from larry.common.config.data.data_configs import TextSourceConfig
from larry.common.training.tokenizer_training import SourceTextReader


class TextSourceTextReader(SourceTextReader[TextSourceConfig]):
    unit = "characters"

    def size(self, source: TextSourceConfig, index: HFDataset, data: HFDataset | None, texts: list[str]) -> float:
        return float(sum(len(t) for t in texts))
