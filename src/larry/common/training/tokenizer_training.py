import logging
from pathlib import Path
from typing import Any

import sentencepiece
from datasets import load_dataset

from larry.common.config.training import build_trainer_config
from larry.common.training.training import Training
from larry.common.utils import load_yaml_config
from larry.voice.config.training.voice_trainer_configs import SpeechTokenizerTrainerConfig


class TokenizerTrainer:
    def __init__(self, config_file: Path, output_file: Path = Path("/tmp/sp_train.txt")) -> None:
        self.log = logging.getLogger(type(self).__name__)
        self.output_file = output_file
        self.config_file = config_file

        definition = load_yaml_config(config_file)
        self.log.info("(Tokenizer) Training config loaded.")

        trainer_config_dict = definition.get("trainer", {})
        self.trainer_config = build_trainer_config(trainer_config_dict)

    def run(self) -> None:
        if isinstance(self.trainer_config, SpeechTokenizerTrainerConfig):
            with open(self.output_file, "w") as f:
                for source in self.trainer_config.mixture_config.train_sources:
                    index = load_dataset(source.index_path, split=source.split)
                    for text in index[source.text_column]:
                        if text:
                            f.write(text + "\n")
                sentencepiece.SentencePieceTrainer.train()
