from larry.common.data.preprocessing.pruners import Pruner
from larry.text.config.preprocessing.text_pruner_configs import TextPrunerConfig


class TextPruner[I, C: TextPrunerConfig = TextPrunerConfig](Pruner[I, C]):
    ...
