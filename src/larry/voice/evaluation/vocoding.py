import torch

from larry.common.model.pretrained import load_pretrained
from larry.voice.config.evaluation.evaluation_configs import PretrainedVocoderConfig
from larry.voice.config.preprocessing.voice_configs import MelsConfig
from larry.voice.model.vocoder import LarryVocoder, LarryVocoderGeneration

MEL_SETTINGS = ("sample_rate", "n_mels", "n_fft", "hop_length")


class MelVocoder:
    def __init__(
            self,
            vocoder: LarryVocoder,
            steps: int,
            seed: int = 0,
            sway_coefficient: float | None = -1.0,
    ) -> None:
        self.vocoder = vocoder
        self.steps = steps
        self.seed = seed
        self.sway_coefficient = sway_coefficient

    @classmethod
    def from_config(cls, config: PretrainedVocoderConfig, device: torch.device) -> MelVocoder:
        vocoder = load_pretrained(LarryVocoder, config.model_path, config.revision).to(device).eval()
        vocoder.requires_grad_(False)
        return cls(vocoder, config.steps, config.seed, config.sway_coefficient)

    @property
    def sample_rate(self) -> int:
        return self.vocoder.config.sample_rate

    @property
    def frame_seconds(self) -> float:
        return self.vocoder.config.hop_length / self.vocoder.config.sample_rate

    def check_compatible(self, mels: MelsConfig) -> None:
        mismatched = {
            name: (getattr(mels, name), getattr(self.vocoder.config, name))
            for name in MEL_SETTINGS
            if getattr(mels, name) != getattr(self.vocoder.config, name)
        }
        if mismatched:
            raise ValueError(f"vocoder expects other mels (given, vocoder): {mismatched}")

    def frontend(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vocoder.frontend(waveforms, waveform_lengths)

    @torch.no_grad()
    def generate(self, mels: torch.Tensor, mel_lengths: torch.Tensor, steps: int | None = None) -> LarryVocoderGeneration:
        generator = torch.Generator(device=mels.device).manual_seed(self.seed)
        return self.vocoder.generate(
            mels,
            mel_lengths,
            steps=self.steps if steps is None else steps,
            generator=generator,
            sway_coefficient=self.sway_coefficient,
        )

    def clips(self, mels: torch.Tensor, mel_lengths: torch.Tensor) -> list[torch.Tensor]:
        return self.generate(mels, mel_lengths).clips()
