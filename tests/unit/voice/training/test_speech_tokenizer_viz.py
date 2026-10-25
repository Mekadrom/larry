from pathlib import Path
from types import SimpleNamespace

import torch

from larry.voice.data.dataloading.voice_dataloading import SpeechCTCCollator
from larry.voice.training.voice_trainers import SpeechEncoderTrainer
from larry.voice.training.voice_viz_callbacks import SpeechEncoderVizCallback, pick_rows
from larry.voice.utils import collapse_ctc_emissions
from tests.helpers.common import uninitialized
from tests.helpers.metrics import RecordingMetrics
from tests.helpers.voice import LETTERS, CharTokenizer, clip_dataset, tiny_speech_encoder, toy_source


def test_frame_seconds_matches_the_encoder_frame_rate() -> None:
    model = tiny_speech_encoder(len(LETTERS)).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 32000), torch.tensor([32000]))

    assert model.encoder1.frame_seconds == 0.04
    assert abs(int(output.frame_lengths[0]) * model.encoder1.frame_seconds - 2.0) <= model.encoder1.frame_seconds + 1e-9


def test_rows_are_spread_over_each_source_and_never_repeat() -> None:
    assert pick_rows([0, 10, 11, 31], per_source=2) == [0, 5, 10, 11, 21]


def test_greedy_frames_collapse_and_time_each_new_token() -> None:
    blank = 0
    frames = torch.tensor([0, 3, 3, 0, 3, 5, 5, 0])
    callback = uninitialized(SpeechEncoderVizCallback, tokenizer=CharTokenizer(), blank=blank)

    assert collapse_ctc_emissions(frames, blank) == [3, 3, 5]
    assert callback._timeline(frames, 0.04) == "`0.04` C `0.16` C `0.20` E"
    assert callback._timeline(None, 0.04) == "(no alignment)"


def test_viz_logs_audio_once_and_alignments_for_every_example_each_time(tmp_path: Path) -> None:
    dataset = clip_dataset([("HI", 1.0), ("OK GO", 1.5)])
    tokenizer = CharTokenizer()
    model = tiny_speech_encoder(len(LETTERS))
    metrics = RecordingMetrics(str(tmp_path))
    trainer = uninitialized(
        SpeechEncoderTrainer,
        accelerator=SimpleNamespace(
            is_main_process=True,
            device=torch.device("cpu"),
            unwrap_model=lambda m: m,
            autocast=torch.no_grad,
        ),
        eval_loader=SimpleNamespace(dataset=dataset),
        make_collate_fn=lambda: SpeechCTCCollator(tokenizer),
        _model=model,
        tokenizer=tokenizer,
        unwrapped_model=model,
        _config=SimpleNamespace(mixture_config=SimpleNamespace(eval_sources=[toy_source()], sample_rate=16000)),
        metrics=metrics,
    )
    callback = SpeechEncoderVizCallback(trainer, viz_steps=10, blank=len(LETTERS), tokenizer=tokenizer)

    callback.viz(0)
    callback.viz(10)

    assert metrics.audio == [("viz/0/toy/audio", 0), ("viz/1/toy/audio", 0)]
    assert [(tag, step) for tag, _, step in metrics.text] == [
        ("viz/0/toy/ctc", 0), ("viz/1/toy/ctc", 0), ("viz/0/toy/ctc", 10), ("viz/1/toy/ctc", 10),
    ]
    assert len(metrics.figures) == 4
    assert "**target:** OK GO" in metrics.text[1][1]
    assert model.training
