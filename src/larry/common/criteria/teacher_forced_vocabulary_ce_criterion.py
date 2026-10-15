import torch
from torch import nn
from torch.nn import Module

from larry.common.config.criteria.criterion_configs import TeacherForcedVocabularyCrossEntropyCriterionConfig
from larry.common.criteria.criterion import Criterion
from larry.common.model.model import LarryModelOutput


class TeacherForcedVocabularyCrossEntropyCriterion[M: Module](
    Criterion[TeacherForcedVocabularyCrossEntropyCriterionConfig, M]
):
    """Cross-Entropy criteria for teacher-forced, causal targets on a sequence of inputs."""

    def __init__(self, config: TeacherForcedVocabularyCrossEntropyCriterionConfig, model: M) -> None:
        """Initializes the backing ``nn.CrossEntropyLoss`` instance with configuration from ``config``."""
        super().__init__(config, model)
        self.ce = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing, ignore_index=-100)

    def forward(self, model_output: LarryModelOutput, **batch_kwargs) -> torch.Tensor:
        """Forward pass for this simple criteria.

        Requires a model output containing raw logits (pre-softmax) and ``batch_kwargs`` containing a ``labels`` tensor,
        then uses ``self.ce`` (an instance of ``nn.CrossEntropyLoss``) to return a reduced loss for causal
        teacher-forcing tasks.
        """
        logits = model_output.logits
        labels = batch_kwargs.get("labels", None)

        if not isinstance(labels, torch.Tensor):
            raise ValueError(
                f"Labels not received when required for criterion={type(self).__name__}, "
                f"received instead: batch_kwargs.keys()={batch_kwargs.keys()}"
            )

        return self.ce(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1))
