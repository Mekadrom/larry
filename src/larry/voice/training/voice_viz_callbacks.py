from abc import ABC

import larry.common.training.trainers as trainers


class VoiceVizCallback(trainers.VizCallback, ABC):
    ...


class SpeechTokenizerVizCallback(VoiceVizCallback):
    def viz(self, trainer: trainers.TrainerBase, step) -> None:
        ...
