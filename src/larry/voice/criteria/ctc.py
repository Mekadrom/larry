import torch
import torch.nn.functional as F

from larry.common.config.criteria.criterion_configs import CTCCriterionConfig
from larry.common.criteria.criterion import Criterion
from larry.common.model.model import LarrySpeechTokenizerModelOutput
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel


class CTCCriterion(Criterion[CTCCriterionConfig, SpeechTokenizerModel]):
    def forward(
            self,
            model_output: LarrySpeechTokenizerModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
            **batch_kwargs,
    ) -> torch.Tensor:
        return F.ctc_loss(
            model_output.ctc_log_probs.transpose(0, 1),
            targets,
            model_output.frame_lengths,
            target_lengths,
            blank=self.config.blank,
            zero_infinity=self.config.zero_infinity,
        )
