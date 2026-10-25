from pathlib import Path

import pytest
import sentencepiece

from larry.multimodal.config.data.dataloading.dataloading_configs import PretrainingMultimodalCollatorConfig
from larry.multimodal.config.processor_configs import LarryProcessorConfig
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.processor.processor import LarryMultimodalPretrainProcessor


@pytest.fixture(scope="module")
def collator(tmp_path_factory: pytest.TempPathFactory) -> PretrainingMultimodalCollator:
    root: Path = tmp_path_factory.mktemp("model")
    (root / "text.txt").write_text("\n".join(["hello there friend", "say this out loud", "a cat on a mat"] * 40))
    sentencepiece.SentencePieceTrainer.train(
        input=str(root / "text.txt"),
        model_prefix=str(root / "sp"),
        vocab_size=24,
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        minloglevel=2,
    )
    config = LarryProcessorConfig(text_tokenizer_model_file_path=str(root / "sp.model"))
    LarryMultimodalPretrainProcessor(config).save_pretrained(root / "processor")
    collator_config = PretrainingMultimodalCollatorConfig(processor_model_path=str(root / "processor"))
    return PretrainingMultimodalCollator(collator_config)
