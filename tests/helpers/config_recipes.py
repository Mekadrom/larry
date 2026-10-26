from pathlib import Path

import yaml

from larry.common.config.config import ConfigValues
from larry.common.config.data.preprocessing.pipeline_config import PipelineDefinition
from larry.common.config.training.definition import TrainingDefinition

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

ENCODER = ["training/voice/mixtures/asr.yaml", "training/voice/ctc/larry-speech-encoder.yaml"]
DECODER = ["training/voice/mixtures/tts.yaml", "training/voice/tts/decoder/larry-speech-decoder.yaml"]
IMAGE_VAE = ["training/image/mixtures/vae.yaml", "training/image/vae/larry-image-vae.yaml"]
VOCODER = ["training/voice/mixtures/vocoder.yaml", "training/voice/vocoder/larry-vocoder.yaml"]
MULTIMODAL = "training/multimodal/larry-multimodal.yaml"
MULTIMODAL_TASKS = ("text-only", "text-voice", "voice-text")
TRAINING_RECIPES: dict[str, list[str]] = {
    "larry-speech-encoder": ENCODER,
    **{
        f"larry-speech-encoder+{overlay.stem}": [*ENCODER, str(overlay.relative_to(CONFIG_ROOT))]
        for overlay in sorted((CONFIG_ROOT / "training/voice/ctc/overlays").glob("*.yaml"))
    },
    **{
        f"larry-speech-encoder+{overlay.stem}": [*ENCODER, str(overlay.relative_to(CONFIG_ROOT))]
        for overlay in sorted((CONFIG_ROOT / "training/common/overlays").glob("*.yaml"))
    },
    "larry-speech-decoder": DECODER,
    **{
        f"larry-speech-decoder+{overlay.stem}": [*DECODER, str(overlay.relative_to(CONFIG_ROOT))]
        for overlay in sorted((CONFIG_ROOT / "training/voice/tts/decoder/overlays").glob("*.yaml"))
    },
    "larry-vocoder": VOCODER,
    **{
        f"larry-vocoder+{overlay.stem}": [*VOCODER, str(overlay.relative_to(CONFIG_ROOT))]
        for overlay in sorted((CONFIG_ROOT / "training/voice/vocoder/overlays").glob("*.yaml"))
    },
    "larry-image-vae": IMAGE_VAE,
    **{
        f"larry-image-vae+{overlay.stem}": [*IMAGE_VAE, str(overlay.relative_to(CONFIG_ROOT))]
        for overlay in sorted((CONFIG_ROOT / "training/image/vae/overlays").glob("*.yaml"))
    },
    "larry-multimodal": [
        *(f"training/multimodal/mixtures/{task}.yaml" for task in MULTIMODAL_TASKS),
        MULTIMODAL,
    ],
    **{
        f"larry-multimodal+{task}": [f"training/multimodal/mixtures/{task}.yaml", MULTIMODAL,
                                     f"training/multimodal/overlays/{task}.yaml"]
        for task in MULTIMODAL_TASKS
    },
    "larry-multimodal+voice-viz": [
        *(f"training/multimodal/mixtures/{task}.yaml" for task in MULTIMODAL_TASKS),
        MULTIMODAL,
        "training/multimodal/overlays/voice-viz.yaml",
    ],
}


def _asr_base(rel: str) -> str:
    # overlays under tts/ and vocoder/ build on the ASR index pipeline of the same dataset
    for variant in ("tts", "vocoder"):
        marker = f"-index-{variant}"
        if f"/{variant}/" in rel and marker in rel:
            return rel.replace(f"/{variant}/", "/asr/").replace(marker, "-index-asr")
    raise ValueError(f"{rel} has no dataset type, so it must be an overlay on an ASR index pipeline")


def _preprocessing_recipes() -> dict[str, list[str]]:
    recipes = {}
    for path in sorted((CONFIG_ROOT / "preprocessing").rglob("*.yaml")):
        rel = str(path.relative_to(CONFIG_ROOT))
        is_overlay = "type" not in (yaml.safe_load(path.read_text()).get("dataset") or {})
        recipes[rel] = [_asr_base(rel), rel] if is_overlay else [rel]
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
