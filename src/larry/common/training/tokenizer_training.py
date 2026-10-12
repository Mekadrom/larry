from pathlib import Path

from datasets import load_dataset

from larry.common.training.training import Training
from larry.voice.config.training.voice_trainer_configs import SpeechTokenizerTrainerConfig


class TokenizerTrainer:
    def __init__(self, output_file: Path, config_file: Path) -> None:
        self.output_file = output_file
        self.config_file = config_file
        training = Training(config_file, Path("/dev/null"))
        self.trainer_config = training.trainer_config

    def run(self) -> None:
        if isinstance(self.trainer_config, SpeechTokenizerTrainerConfig):
            with open("sp_train.txt", "w") as f:
                for source in self.trainer_config.mixture_config.train_sources:
                    index = load_dataset(source.index_path, split=source.split)
                    for text in index[source.text_column]:
                        if text:
                            f.write(text + "\n")
