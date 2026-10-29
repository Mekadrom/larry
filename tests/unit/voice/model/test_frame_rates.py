import pytest
import torch

from larry.voice.config.model.voice_model_configs import LarryVocoderConfig
from larry.voice.model.vocoder import LarryVocoder
from tests.helpers.voice import LETTERS, tiny_speech_encoder, tiny_speech_encoder_config
from tests.unit.voice.model.test_speech_decoder import _model as tiny_decoder


def test_two_seconds_of_audio_become_twenty_five_units() -> None:
    model = tiny_speech_encoder(len(LETTERS)).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 32000) * 0.1, torch.tensor([32000]))

    assert int(output.frame_lengths[0]) == 25
    assert output.unit_ids.shape[1] == output.logits.shape[1] >= 25
    assert torch.equal(output.ctc_lengths, output.frame_lengths)


def test_ctc_upsampling_gives_more_ctc_frames_but_the_same_units() -> None:
    model = tiny_speech_encoder(len(LETTERS), ctc_upsample=2).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 32000) * 0.1, torch.tensor([32000]))

    frames = output.unit_ids.shape[1]
    assert int(output.frame_lengths[0]) == 25
    assert output.logits.shape == (1, 2 * frames, len(LETTERS) + 1)
    assert torch.equal(output.ctc_lengths, 2 * output.frame_lengths)
    assert torch.allclose(output.logits.exp().sum(-1), torch.ones(1, 2 * frames))


def test_padding_does_not_change_a_shorter_clips_units() -> None:
    model = tiny_speech_encoder(len(LETTERS)).eval()
    torch.manual_seed(1)
    waveform = torch.randn(17_000) * 0.1
    padded = torch.stack([torch.cat([waveform, torch.zeros(15_000)]), torch.randn(32_000) * 0.1])

    with torch.no_grad():
        alone = model.tokenize(waveform[None], torch.tensor([17_000]))
        batched = model.tokenize(padded, torch.tensor([17_000, 32_000]))

    n = int(alone[1][0])
    assert int(batched[1][0]) == n
    assert torch.equal(batched[0][0, :n], alone[0][0, :n])


def test_merged_layers_use_the_shorter_fsmn_kernel() -> None:
    model = tiny_speech_encoder(len(LETTERS), fsmn_kernel_size=31, merged_fsmn_kernel_size=15)

    assert [b.attn.fsmn.kernel_size[0] for b in model.encoder1.blocks] == [31]
    assert [b.attn.fsmn.kernel_size[0] for b in model.encoder1.merged_blocks] == [15]


def test_frame_merge_of_one_keeps_the_conv_frame_rate() -> None:
    model = tiny_speech_encoder(len(LETTERS), frame_merge=1).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 32000) * 0.1, torch.tensor([32000]))

    assert model.encoder1.merge is None
    assert int(output.frame_lengths[0]) == 50


def test_more_premerge_layers_than_encoder1_layers_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_premerge_layers"):
        tiny_speech_encoder(len(LETTERS), n_premerge_layers=3)


def test_the_encoder_unit_rate_matches_the_decoders_mel_rate() -> None:
    encoder = tiny_speech_encoder_config(len(LETTERS))
    decoder = tiny_decoder().config
    encoder_frame = encoder.hop_length * 2 * 2 * encoder.frame_merge / encoder.sample_rate

    assert encoder_frame == decoder.hop_length * decoder.unit_upsample / decoder.sample_rate == 0.08


def test_unit_conditioning_is_upsampled_to_mel_frames_and_ignores_padded_units() -> None:
    decoder = tiny_decoder().eval()
    unit_ids = torch.randint(0, 81, (2, 13), generator=torch.Generator().manual_seed(0))
    lengths = torch.tensor([13, 9])
    changed = unit_ids.clone()
    changed[1, 9:] = 80

    with torch.no_grad():
        before = decoder.unit_encoder(unit_ids, lengths)
        after = decoder.unit_encoder(changed, lengths)

    assert before.shape == (2, decoder.config.n_mels, 13 * 4)
    assert torch.equal(before[1, :, :9 * 4], after[1, :, :9 * 4])


def test_the_vocoder_generates_its_stft_at_twice_the_mel_rate() -> None:
    torch.manual_seed(0)
    model = LarryVocoder(LarryVocoderConfig(d_model=32, d_inner=64, n_layers=2)).eval()
    waveforms = torch.randn(2, 16_000) * 0.1
    mels, mel_lengths = model.frontend(waveforms, torch.tensor([16_000, 12_000]))

    with torch.no_grad():
        out = model.generate(mels, mel_lengths, steps=2, generator=torch.Generator().manual_seed(0))

    assert model.mel_upsample == 2
    assert out.waveforms.shape[-1] == (2 * mels.shape[-1] - 1) * model.config.stft_hop_length
    assert torch.isfinite(out.waveforms).all()


def test_a_mel_hop_that_is_not_a_multiple_of_the_stft_hop_is_rejected() -> None:
    with pytest.raises(ValueError, match="stft_hop_length"):
        LarryVocoder(LarryVocoderConfig(d_model=32, d_inner=64, n_layers=2, hop_length=240))
