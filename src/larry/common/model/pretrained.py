import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from safetensors.torch import load_model

from larry.common.config.model.model_configs import ModelConfig

if TYPE_CHECKING:
    from larry.common.training.training_model import LarryModel

MODEL_CONFIG_FILE = "config.json"
MODEL_WEIGHTS_FILE = "model.safetensors"


def training_checkpoint_config(path: Path) -> Path | None:
    if not (path / MODEL_WEIGHTS_FILE).is_file() or (path / MODEL_CONFIG_FILE).is_file():
        return None

    config_file = path.parent / MODEL_CONFIG_FILE
    return config_file if config_file.is_file() else None


def load_pretrained[M: LarryModel[Any]](model_type: type[M], path: str, revision: str | None = None) -> M:
    local = Path(path).expanduser()
    config_file = training_checkpoint_config(local)
    if config_file is None:
        return cast("Any", model_type).from_pretrained(path, revision=revision)

    config = ModelConfig.from_dict(json.loads(config_file.read_text()))
    model = model_type(config)
    load_model(model, local / MODEL_WEIGHTS_FILE)
    return model
