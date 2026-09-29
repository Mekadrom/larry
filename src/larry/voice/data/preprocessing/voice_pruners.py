import torch
from torchcodec.decoders import AudioDecoder

from larry.common.data.preprocessing.pruners import Pruner
from larry.utils.types import EncodedAudio
from larry.voice.config.voice_pruner_configs import AudioDurationColumnPrunerConfig, AudioQualityPrunerConfig, \
    VoicePrunerConfig


class VoicePruner[I, C=VoicePrunerConfig](Pruner[I, C]):
    ...


class AudioDurationColumnPruner(VoicePruner[int | float, AudioDurationColumnPrunerConfig]):
    def prune_example(self, example: int | float) -> bool:
        return example < self.config.min_duration or example > self.config.max_duration


class AudioQualityPruner(VoicePruner[EncodedAudio, AudioQualityPrunerConfig]):
    def prune_example(self, example: EncodedAudio) -> bool:
        samples = AudioDecoder(example["bytes"], sample_rate=self.config.sample_rate).get_all_samples()
        waveform = samples.data

        if waveform.ndim > 1 and waveform.shape[0] == 2:
            # downmix to mono
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.ndim > 1 and waveform.shape[0] == 1:
            # remove single channel
            waveform = waveform.squeeze(0)

        frame, hop = int(0.025 * self.config.sample_rate), int(0.010 * self.config.sample_rate)

        if waveform.shape[-1] < frame:
            return True

        db = 10 * torch.log10(waveform.unfold(0, frame, hop).pow(2).mean(-1).clamp(min=1e-10))
        active = db > db.max() - 35
        noise_floor = torch.quantile(db, 0.10)
        speech_level = torch.quantile(db[active], 0.50)

        rms_dbfs = (10 * torch.log10(waveform.pow(2).mean().clamp(min=1e-10))).item()
        active_ratio = active.float().mean().item()
        snr_db = (speech_level - noise_floor).item()
        clip_frac = (waveform.abs() >= 0.999).float().mean().item()

        prune_rms_dbfs = rms_dbfs < self.config.min_rms_dbfs
        prune_active_ratio = active_ratio < self.config.min_active_speech_ratio
        prune_snr = snr_db < self.config.min_snr
        prune_clip_frac = clip_frac > self.config.max_clipping_portion

        return any([prune_rms_dbfs, prune_active_ratio, prune_snr, prune_clip_frac])
