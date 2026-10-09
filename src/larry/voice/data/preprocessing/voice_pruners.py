import torch

from larry.common.data.preprocessing.pruners import Pruner
from larry.common.utils.types import EncodedAudio
from larry.voice.config.preprocessing.voice_pruner_configs import AudioDurationColumnPrunerConfig, \
    AudioQualityPrunerConfig, VoicePrunerConfig
from larry.voice.data.preprocessing.alignment import Aligner
from larry.voice.utils import bytes_to_waveforms

_ALIGNERS: dict[str, Aligner] = {}


class VoicePruner[I, C: VoicePrunerConfig = VoicePrunerConfig](Pruner[I, C]):
    ...


class AudioDurationColumnPruner(VoicePruner[int | float, AudioDurationColumnPrunerConfig]):
    def keep_example(self, example: int | float) -> bool:
        return example < self.config.min_duration or example > self.config.max_duration


class AudioQualityPruner(VoicePruner[EncodedAudio, AudioQualityPrunerConfig]):
    def keep_example(self, example: EncodedAudio) -> bool:
        waveform = bytes_to_waveforms(example["bytes"], self.config.sample_rate)

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
