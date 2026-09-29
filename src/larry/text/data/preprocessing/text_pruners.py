from larry.common.data.preprocessing.pruners import Pruner
from larry.text.config.text_pruner_configs import TextPrunerConfig


class TextPruner[I, C=TextPrunerConfig](Pruner[I, C]):
    ...
