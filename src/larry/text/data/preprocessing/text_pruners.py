from larry.common.data.preprocessing.pruners import Pruner
from larry.text.config.preprocessing.text_pruner_configs import TextPrunerConfig


class TextPruner[C: TextPrunerConfig = TextPrunerConfig](Pruner[C]):
    """Superclass for text pruners."""

    ...
