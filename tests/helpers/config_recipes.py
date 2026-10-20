from pathlib import Path

from larry.common.config.config import ConfigValues
from larry.common.config.data.preprocessing.pipeline_config import PipelineDefinition
from larry.common.config.training.definition import TrainingDefinition

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

ENCODER = ["training/voice/mixtures/asr.yaml", "training/voice/ctc/larry-speech-encoder.yaml"]
TRAINING_RECIPES: dict[str, list[str]] = {
    "larry-speech-encoder": ENCODER,
    **{
        f"larry-speech-encoder+{overlay.stem}": [*ENCODER, str(overlay.relative_to(CONFIG_ROOT))]
        for overlay in sorted((CONFIG_ROOT / "training/voice/ctc/overlays").glob("*.yaml"))
    },
    "larry-speech-decoder": ["training/voice/mixtures/tts.yaml",
                             "training/voice/tts/decoder/larry-speech-decoder.yaml"],
}


def _preprocessing_recipes() -> dict[str, list[str]]:
    recipes = {}
    for path in sorted((CONFIG_ROOT / "preprocessing").rglob("*.yaml")):
        rel = str(path.relative_to(CONFIG_ROOT))
        head, _, tail = rel.rpartition("-tts")
        base = (head + "-asr" + tail).replace("/tts/", "/asr/")
        recipes[rel] = [base, rel] if "/tts/" in rel and (CONFIG_ROOT / base).exists() else [rel]
    return recipes


PREPROCESSING_RECIPES = _preprocessing_recipes()


def values(files: list[str]) -> ConfigValues:
    return ConfigValues([CONFIG_ROOT / f for f in files])


def load_training(name: str) -> TrainingDefinition:
    return values(TRAINING_RECIPES[name]).load(TrainingDefinition)


def load_pipeline(name: str) -> PipelineDefinition:
    return values(PREPROCESSING_RECIPES[name]).load(PipelineDefinition)


def load_pipelines() -> dict[str, PipelineDefinition]:
    return {name: load_pipeline(name) for name in PREPROCESSING_RECIPES}
