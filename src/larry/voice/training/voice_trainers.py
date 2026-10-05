from abc import ABC
from typing import Any, Mapping

from larry.common.config.criteria.criterion_configs import TeacherForcedVocabularyCrossEntropyCriterionConfig
from larry.common.criteria import criterion
from larry.common.criteria.teacher_forced_vocabulary_ce_criterion import TeacherForcedVocabularyCrossEntropyCriterion
from larry.common.training.trainers import LarryTrainer, VizCallback
from larry.voice.config.training.voice_trainer_configs import VoiceTrainerConfig, SpeechTokenizerConfig
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel
from larry.voice.model.voice_models import VoiceModel
from larry.voice.training.voice_viz_callbacks import SpeechTokenizerVizCallback


class VoiceTrainer[M: VoiceModel, C: VoiceTrainerConfig = VoiceTrainerConfig](LarryTrainer[M, C], ABC):
    def batch_size(self, batch: Mapping[str, Any]) -> int:
        return len(batch["audio"])


class SpeechTokenizerTrainer(VoiceTrainer[SpeechTokenizerModel, SpeechTokenizerConfig]):
    def make_criterion(self) -> criterion.Criterion:
        return TeacherForcedVocabularyCrossEntropyCriterion(TeacherForcedVocabularyCrossEntropyCriterionConfig())

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        return SpeechTokenizerVizCallback(viz_steps)
