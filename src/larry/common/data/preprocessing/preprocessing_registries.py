from larry.common.config import mapper_configs, pruner_configs, preprocessor_configs
from larry.common.config.dataset_config import DatasetConfig
from larry.common.config.mapper_configs import MapperConfig
from larry.common.config.preprocessor_configs import PreprocessorConfig
from larry.common.config.pruner_configs import PrunerConfig
from larry.common.data import dataset_providers
from larry.common.data.dataset_providers import DatasetProvider
from larry.common.data.preprocessing import Mapper, mappers, pruners, preprocessors
from larry.common.data.preprocessing.preprocessors import Preprocessor
from larry.image.config import image_mapper_configs, image_dataset_configs, image_pruner_configs
from larry.image.config.image_mapper_configs import ImageMapperConfig
from larry.image.config.image_pruner_configs import ImagePrunerConfig
from larry.image.data import image_mappers, image_dataset_providers, image_pruners
from larry.text.config import text_mapper_configs, text_dataset_configs, text_pruner_configs
from larry.text.config.text_mapper_configs import TextMapperConfig
from larry.text.config.text_pruner_configs import TextPrunerConfig
from larry.text.data import text_dataset_providers
from larry.text.data.preprocessing import text_mappers, text_pruners
from larry.utils.class_inspector import get_classes
from larry.utils.types import FactoryRegistry, StringRegistry, Modality
from larry.voice.config import voice_mapper_configs, voice_dataset_configs, voice_pruner_configs
from larry.voice.config.voice_mapper_configs import VoiceMapperConfig
from larry.voice.config.voice_pruner_configs import VoicePrunerConfig
from larry.voice.data import voice_dataset_providers
from larry.voice.data.preprocessing import voice_mappers, voice_pruners

dataset_provider_config_classes_by_modality: StringRegistry[FactoryRegistry[DatasetConfig]] = {}
dataset_provider_classes_by_name: FactoryRegistry[DatasetProvider] = {}
preprocessor_config_classes: FactoryRegistry[PreprocessorConfig] = {}
preprocessor_classes: FactoryRegistry[Preprocessor] = {}
default_preprocessor_config_classes_by_mapper_class: FactoryRegistry[PreprocessorConfig] = {}


def register_provider_config_classes_by_modality(modality: Modality | str, classes: dict[str, type]) -> None:
    dataset_provider_config_classes_by_modality[modality] = classes


register_provider_config_classes_by_modality("voice", get_classes(voice_dataset_configs))
register_provider_config_classes_by_modality("text", get_classes(text_dataset_configs))
register_provider_config_classes_by_modality("image", get_classes(image_dataset_configs))


def register_provider_classes_by_name(classes: dict[str, type]) -> None:
    dataset_provider_classes_by_name.update(classes)


[register_provider_classes_by_name(get_classes(module)) for module in [
    voice_dataset_providers,
    text_dataset_providers,
    image_dataset_providers,
    dataset_providers,
]]


def register_preprocessor_config_classes(classes: FactoryRegistry[PreprocessorConfig]) -> None:
    preprocessor_config_classes.update(classes)


[register_preprocessor_config_classes(get_classes(module)) for module in [
    voice_mapper_configs,
    text_mapper_configs,
    image_mapper_configs,
    mapper_configs,
    voice_pruner_configs,
    text_pruner_configs,
    image_pruner_configs,
    pruner_configs,
    preprocessor_configs
]]


def register_default_preprocessor_config_classes_by_preprocessor_class(
        mappings: FactoryRegistry[PreprocessorConfig]) -> None:
    default_preprocessor_config_classes_by_mapper_class.update(mappings)


def register_preprocessor_classes(default_config_type: type, classes: FactoryRegistry[Mapper]) -> None:
    register_default_preprocessor_config_classes_by_preprocessor_class({
        c: default_config_type
        for c in classes.keys()
    })
    preprocessor_classes.update(classes)


[register_preprocessor_classes(default_config, get_classes(module)) for default_config, module in [
    (VoiceMapperConfig, voice_mappers),
    (TextMapperConfig, text_mappers),
    (ImageMapperConfig, image_mappers),
    (MapperConfig, mappers),
    (VoicePrunerConfig, voice_pruners),
    (TextPrunerConfig, text_pruners),
    (ImagePrunerConfig, image_pruners),
    (PrunerConfig, pruners),
    (PreprocessorConfig, preprocessors),
]]
