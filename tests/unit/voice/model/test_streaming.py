import pytest
import torch

from larry.voice.model.speech_decoder import SpeechDecoderModel
from larry.voice.model.vocoder import LarryVocoder
from tests.unit.voice.model.test_speech_decoder import _batch
from tests.unit.voice.model.test_speech_decoder import _model as tiny_decoder
from tests.unit.voice.model.test_vocoder import TINY

UPSAMPLE = 4
CHUNK = 6


def _active_decoder(**overrides: object) -> SpeechDecoderModel:
    model = tiny_decoder(**overrides).eval()
    torch.nn.init.normal_(model.backbone.out.weight, std=0.1)
    return model


def _prosody(frames: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    contour = torch.randn(1, frames, generator=generator)
    voiced = (torch.rand(1, frames, generator=generator) > 0.3).float()
    aperiodicity = torch.rand(1, frames, 8, generator=generator)
    return contour, voiced, aperiodicity


@pytest.mark.parametrize(
    "overrides",
    [{}, {"infill_enabled": True}, {"harmonic_mode": "sinusoid"}],
    ids=["comb", "infill", "sinusoid"],
)
@pytest.mark.parametrize("cfg_scale", [1.0, 2.0])
def test_streamed_mels_match_offline_chunked_generation(overrides: dict[str, object], cfg_scale: float) -> None:
    model = _active_decoder(**overrides)
    batch = _batch()
    unit_ids, speakers = batch["unit_ids"][:1], batch["speaker_ids"][:1]
    frames = unit_ids.shape[1] * UPSAMPLE
    contour, voiced, aperiodicity = _prosody(frames)
    noise = torch.randn(1, 80, frames, generator=torch.Generator().manual_seed(1))

    offline = model.generate(
        unit_ids, torch.tensor([unit_ids.shape[1]]), speakers, steps=3, cfg_scale=cfg_scale, chunk_units=CHUNK,
        contour=contour, voiced=voiced, aperiodicity=aperiodicity, noise=noise,
    ).mels

    stream = model.stream(speakers, 1, steps=3, chunk_units=CHUNK, cfg_scale=cfg_scale)
    pieces = []
    for start in range(0, unit_ids.shape[1], CHUNK):
        end = min(start + CHUNK, unit_ids.shape[1])
        frame_slice = slice(start * UPSAMPLE, end * UPSAMPLE)
        pieces.append(stream.push(
            unit_ids[:, start:end], contour[:, frame_slice], voiced[:, frame_slice], aperiodicity[:, frame_slice],
            noise[..., frame_slice],
        ))

    streamed = torch.cat(pieces, dim=-1)
    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-2 if overrides.get("harmonic_mode") == "sinusoid" else 1e-4)


def test_streams_reject_oversized_chunks_and_pushes_after_a_short_one() -> None:
    model = _active_decoder()
    unit_ids = _batch()["unit_ids"][:1]
    stream = model.stream(None, 1, steps=2, chunk_units=CHUNK)

    with pytest.raises(ValueError):
        stream.push(unit_ids[:, :CHUNK + 1])

    stream.push(unit_ids[:, :CHUNK - 1])
    with pytest.raises(ValueError):
        stream.push(unit_ids[:, :CHUNK])


def test_a_flat_contour_matches_a_flat_f0_track() -> None:
    model = _active_decoder()
    batch = _batch()
    unit_ids, lengths, speakers = batch["unit_ids"][:1], torch.tensor([25]), batch["speaker_ids"][:1]
    frames = 25 * UPSAMPLE
    _, voiced, aperiodicity = _prosody(frames)
    flat_f0 = 165.0 * voiced
    noise = torch.randn(1, 80, frames, generator=torch.Generator().manual_seed(1))

    def generate(**prosody: torch.Tensor) -> torch.Tensor:
        return model.generate(
            unit_ids, lengths, speakers, steps=2, aperiodicity=aperiodicity, noise=noise, **prosody,
        ).mels

    from_f0 = generate(f0=flat_f0)
    from_contour = generate(contour=torch.zeros(1, frames), voiced=voiced)

    assert torch.allclose(from_f0, from_contour, atol=1e-5)


def test_a_contour_needs_voicing_and_excludes_f0() -> None:
    model = _active_decoder()
    batch = _batch()
    args = (batch["unit_ids"][:1], torch.tensor([25]), None)

    with pytest.raises(ValueError):
        model.generate(*args, steps=1, contour=torch.zeros(1, 100))
    with pytest.raises(ValueError):
        model.generate(*args, steps=1, contour=torch.zeros(1, 100), voiced=torch.ones(1, 100), f0=torch.ones(1, 100))


def _active_vocoder() -> LarryVocoder:
    torch.manual_seed(0)
    vocoder = LarryVocoder(TINY).eval()
    torch.nn.init.normal_(vocoder.backbone.out.weight, std=0.1)
    return vocoder


@pytest.mark.parametrize("sizes", [[7, 5, 13, 15], [1] * 40, [40]])
def test_one_step_streamed_audio_matches_offline_vocoding(sizes: list[int]) -> None:
    vocoder = _active_vocoder()
    mels = torch.randn(1, TINY.n_mels, sum(sizes), generator=torch.Generator().manual_seed(0)) - 3
    frames = mels.shape[-1] * vocoder.mel_upsample
    noise = torch.randn(1, 2 * vocoder.n_bins, frames, generator=torch.Generator().manual_seed(1))
    offline = vocoder.generate(mels, torch.tensor([mels.shape[-1]]), steps=1, noise=noise).waveforms

    margin = vocoder.receptive_field + 1
    stream = vocoder.stream(steps=1, lookahead=margin, left_context=margin)
    pieces, start = [], 0
    for size in sizes:
        end = start + size
        pieces.append(stream.push(mels[..., start:end], noise[..., start * 2:end * 2]))
        start = end

    pieces.append(stream.finish())
    streamed = torch.cat(pieces, dim=-1)

    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-5)


def test_streamed_audio_is_continuous_with_default_settings() -> None:
    vocoder = _active_vocoder()
    mels = torch.randn(1, TINY.n_mels, 60, generator=torch.Generator().manual_seed(0)) - 3
    stream = vocoder.stream(steps=4, generator=torch.Generator().manual_seed(2))

    pieces = [stream.push(mels[..., i:i + 10]) for i in range(0, 60, 10)] + [stream.finish()]

    assert sum(p.shape[-1] for p in pieces) == (60 * 2 - 1) * TINY.stft_hop_length
    assert stream.mels is not None and stream.mels.shape[-1] <= stream.left_context + stream.lookahead
    assert all(torch.isfinite(p).all() for p in pieces)


def test_vocoder_streams_validate_their_windows() -> None:
    vocoder = _active_vocoder()

    with pytest.raises(ValueError):
        vocoder.stream(steps=1, lookahead=1)
    with pytest.raises(ValueError):
        vocoder.stream(steps=1, crossfade=TINY.hop_length + 1)
