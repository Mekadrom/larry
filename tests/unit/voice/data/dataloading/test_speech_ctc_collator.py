import torch
from torch.utils.data import DataLoader

from larry.voice.data.dataloading.voice_dataloading import SpeechCTCCollator


class CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 7 for c in text]


ITEMS = [
    {"waveform": torch.ones(5), "text": "AB", "source": 0},
    {"waveform": torch.ones(3), "text": "CDEF", "source": 2},
    {"waveform": torch.ones(4), "text": "", "source": 0},
]


def test_collator_pads_and_reports_target_token_count_as_a_plain_int() -> None:
    batch = SpeechCTCCollator(CharTokenizer())(ITEMS)

    assert batch["waveforms"].shape == (3, 5)
    assert batch["waveform_lengths"].tolist() == [5, 3, 4]
    assert batch["target_lengths"].tolist() == [2, 4, 0]
    assert batch["source_ids"].tolist() == [0, 2, 0]
    assert batch["n_target_tokens"] == 6
    assert type(batch["n_target_tokens"]) is int


def test_target_token_count_survives_dataloader_worker_processes() -> None:
    loader = DataLoader(ITEMS, batch_size=3, collate_fn=SpeechCTCCollator(CharTokenizer()), num_workers=2)

    assert [batch["n_target_tokens"] for batch in loader] == [6]
