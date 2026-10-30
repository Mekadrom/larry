import contextlib
import dataclasses
from typing import Any

import torch

from larry.common.criteria.criterion import LossTerm
from larry.common.criteria.flow_matching import ConditionalFlowMatchingCriterion
from larry.common.model.masks import make_non_pad_mask
from larry.voice.config.criterion.voice_criterion_configs import SpeechDecoderCriterionConfig, VocoderCriterionConfig
from larry.voice.config.model.voice_model_configs import LarryVocoderConfig
from larry.voice.criteria.spectral import MultiResolutionSTFTLoss
from larry.voice.model.voice_models import LarryVocoderOutput, SpeechDecoderModelOutput
from larry.voice.model.voice_modules import TTSLogMelFrontend


class SpeechDecoderCriterion(ConditionalFlowMatchingCriterion[SpeechDecoderModelOutput, SpeechDecoderCriterionConfig]):
    def kl_scale(self, step: int | None) -> float:
        if step is None or self.config.kl_warmup_steps <= 0:
            return self.config.kl_weight

        return self.config.kl_weight * min(1.0, step / self.config.kl_warmup_steps)

    def prior_term(self, model_output: SpeechDecoderModelOutput) -> LossTerm:
        mask = model_output.frame_mask.unsqueeze(1)
        squared = (model_output.prior - model_output.prior_targets).square() * mask
        # averaged over every valid frame and mel bin in the batch
        counts = mask.sum(dim=(1, 2)) * model_output.prior.shape[1]
        return LossTerm(weight=self.config.prior_weight, sums=squared.sum(dim=(1, 2)), counts=counts)

    def prior_loss(self, model_output: SpeechDecoderModelOutput) -> torch.Tensor:
        return self.prior_term(model_output).value

    def loss_terms(
            self,
            model_output: SpeechDecoderModelOutput,
            step: int | None = None,
            **batch_kwargs: Any,
    ) -> dict[str, LossTerm]:
        kl = model_output.kl
        terms = {
            "flow": self.flow_term(model_output),
            "prior": self.prior_term(model_output),
            "kl": LossTerm(weight=self.kl_scale(step), sums=kl, counts=torch.ones_like(kl)),
        }
        if model_output.f0_range is not None:
            terms["f0_range"] = self.f0_range_term(model_output)

        return terms

    def f0_range_term(self, model_output: SpeechDecoderModelOutput) -> LossTerm:
        assert model_output.f0_range is not None and model_output.f0_range_targets is not None
        assert model_output.f0_range_mask is not None
        mask = model_output.f0_range_mask.to(model_output.f0_range.dtype)
        squared = (model_output.f0_range - model_output.f0_range_targets.to(model_output.f0_range.dtype)).pow(2)
        return LossTerm(weight=self.config.f0_range_weight, sums=squared.sum(dim=1) * mask, counts=2 * mask)


class VocoderCriterion(ConditionalFlowMatchingCriterion[LarryVocoderOutput, VocoderCriterionConfig]):
    def __init__(self, config: VocoderCriterionConfig, model_config: LarryVocoderConfig) -> None:
        super().__init__(config)
        self.frontend = TTSLogMelFrontend(
            model_config.sample_rate, model_config.n_mels, model_config.n_fft, model_config.hop_length
        )
        self.stft = MultiResolutionSTFTLoss(config.stft_fft_sizes)

    @staticmethod
    def gradients_for(weight: float) -> contextlib.AbstractContextManager[Any]:
        return contextlib.nullcontext() if weight != 0 else torch.no_grad()

    def estimate_term(self, weight: float, values: torch.Tensor, t: torch.Tensor) -> LossTerm:
        weighted = values * t.to(values.dtype).pow(self.config.estimate_t_power)
        return LossTerm(weight=weight, sums=weighted, counts=torch.ones_like(weighted))

    def stft_term(
            self,
            model_output: LarryVocoderOutput,
            waveforms: torch.Tensor,
            waveform_lengths: torch.Tensor,
    ) -> LossTerm:
        estimates = model_output.waveform_estimates
        n = estimates.shape[-1]
        with self.gradients_for(self.config.stft_weight):
            values = self.stft(estimates, waveforms[..., :n], waveform_lengths.clamp(max=n))

        return self.estimate_term(self.config.stft_weight, values, model_output.t)

    def mel_term(self, model_output: LarryVocoderOutput, mels: torch.Tensor, mel_lengths: torch.Tensor) -> LossTerm:
        estimates = model_output.waveform_estimates
        with self.gradients_for(self.config.mel_weight):
            estimate_mels, _ = self.frontend(estimates, mel_lengths.new_full(mel_lengths.shape, estimates.shape[-1]))
            frames = min(estimate_mels.shape[-1], mels.shape[-1])
            mask = make_non_pad_mask(mel_lengths.clamp(max=frames), frames).unsqueeze(1).to(estimate_mels.dtype)
            distance = (estimate_mels[..., :frames] - mels[..., :frames].float()).abs() * mask
            values = distance.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * mels.shape[1]).clamp(min=1)

        return self.estimate_term(self.config.mel_weight, values, model_output.t)

    def loss_terms(
            self,
            model_output: LarryVocoderOutput,
            waveforms: torch.Tensor | None = None,
            waveform_lengths: torch.Tensor | None = None,
            mels: torch.Tensor | None = None,
            mel_lengths: torch.Tensor | None = None,
            **batch_kwargs: Any,
    ) -> dict[str, LossTerm]:
        terms = {"flow": dataclasses.replace(self.flow_term(model_output), weight=self.config.flow_weight)}
        if waveforms is not None and waveform_lengths is not None:
            terms["stft"] = self.stft_term(model_output, waveforms, waveform_lengths)

        if mels is not None and mel_lengths is not None:
            terms["mel"] = self.mel_term(model_output, mels, mel_lengths)

        return terms
