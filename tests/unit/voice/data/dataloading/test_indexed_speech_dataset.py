import torch
from datasets import Dataset

from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.data.dataloading.voice_dataloading import SpeechSourceConfig


def test_source_ids_and_durations_cover_every_index_in_order_with_full_precision() -> None:
    a = Dataset.from_dict({"duration_s": [1.25, 2.5], "audio": [b"x", b"y"]})
    b = Dataset.from_dict({"duration_s": [0.1, 3.3, 7.123456789], "audio": [b"x", b"y", b"z"]})
    b = b.filter(lambda row: row["duration_s"] > 0.2)
    dataset = IndexedSpeechDataset(
        [SpeechSourceConfig(name="a", index_path="a"), SpeechSourceConfig(name="b", index_path="b")],
        [a, b],
        [None, None],
        sample_rate=16000,
    )

    source_ids, durations = dataset.source_ids_and_durations()

    assert source_ids.tolist() == [0, 0, 1, 1]
    assert durations.dtype == torch.float64
    assert durations.tolist() == [1.25, 2.5, 3.3, 7.123456789]
