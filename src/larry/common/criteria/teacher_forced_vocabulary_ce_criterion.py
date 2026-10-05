import torch
from torch import nn

from larry.common.config.criteria.criterion_configs import TeacherForcedVocabularyCrossEntropyCriterionConfig
from larry.common.criteria.criterion import Criterion
from larry.common.model.model import LarryModelOutput


class TeacherForcedVocabularyCrossEntropyCriterion(Criterion[TeacherForcedVocabularyCrossEntropyCriterionConfig]):
    def __init__(self, config: TeacherForcedVocabularyCrossEntropyCriterionConfig) -> None:
        super().__init__(config)
        self.ce = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing, ignore_index=-100)

    def forward(self, model_output: LarryModelOutput, **batch_kwargs) -> torch.Tensor:
        logits = model_output.logits
        labels = batch_kwargs.get("labels", None)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(
                f"Labels not received when required for criterion={type(self).__name__}, "
                f"received instead: batch_kwargs.keys()={batch_kwargs.keys()}"
            )

        return self.ce(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1))
