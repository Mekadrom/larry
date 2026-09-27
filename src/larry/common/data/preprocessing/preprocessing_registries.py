from larry.common.config import preprocessing_mapper_configs
from larry.common.config.dataset_config import DatasetConfig
from larry.common.config.preprocessing_mapper_configs import PreprocessingMapperConfig
from larry.common.data.dataset_providers import DatasetProvider
from larry.common.data.preprocessing import PreprocessingMapper, preprocessing_mappers
from larry.image.config import image_preprocessing_mapper_configs, image_dataset_configs
from larry.image.config.image_preprocessing_mapper_configs import ImagePreprocessingMapperConfig
from larry.image.data import image_preprocessing_mappers, image_dataset_providers
from larry.text.config import text_preprocessing_mapper_configs, text_dataset_configs
from larry.text.config.text_preprocessing_mapper_configs import TextPreprocessingMapperConfig
from larry.text.data import text_dataset_providers
from larry.text.data.preprocessing import text_preprocessing_mappers
from larry.utils.class_inspector import get_classes
from larry.utils.types import FactoryRegistry, StringRegistry, Modality
from larry.voice.config import voice_preprocessing_mapper_configs, voice_dataset_configs
from larry.voice.config.voice_preprocessing_mapper_configs import VoicePreprocessingMapperConfig
from larry.voice.data import voice_dataset_providers
from larry.voice.data.preprocessing import voice_preprocessing_mappers

dataset_provider_config_classes_by_modality: StringRegistry[FactoryRegistry[DatasetConfig]] = {}
dataset_provider_classes_by_name: FactoryRegistry[DatasetProvider] = {}
mapper_config_classes: FactoryRegistry[PreprocessingMapperConfig] = {}
mapper_classes: FactoryRegistry[PreprocessingMapper] = {}
default_mapper_config_classes_by_mapper_class: FactoryRegistry[PreprocessingMapperConfig] = {}


def register_provider_config_classes_by_modality(modality: Modality | str, classes: dict[str, type]) -> None:
    dataset_provider_config_classes_by_modality[modality] = classes


register_provider_config_classes_by_modality("voice", get_classes(voice_dataset_configs))
register_provider_config_classes_by_modality("text", get_classes(text_dataset_configs))
register_provider_config_classes_by_modality("image", get_classes(image_dataset_configs))


def register_provider_classes_by_name(classes: dict[str, type]) -> None:
    dataset_provider_classes_by_name.update(classes)


register_provider_classes_by_name(get_classes(voice_dataset_providers))
register_provider_classes_by_name(get_classes(text_dataset_providers))
register_provider_classes_by_name(get_classes(image_dataset_providers))


def register_mapper_config_classes(classes: FactoryRegistry[PreprocessingMapperConfig]) -> None:
    mapper_config_classes.update(classes)


register_mapper_config_classes(get_classes(voice_preprocessing_mapper_configs))
register_mapper_config_classes(get_classes(text_preprocessing_mapper_configs))
register_mapper_config_classes(get_classes(image_preprocessing_mapper_configs))
register_mapper_config_classes(get_classes(preprocessing_mapper_configs))


def register_mapper_classes(classes: FactoryRegistry[PreprocessingMapper]) -> None:
    mapper_classes.update(classes)


voice_mapper_classes: StringRegistry[type] = get_classes(voice_preprocessing_mappers)
text_mapper_classes: StringRegistry[type] = get_classes(text_preprocessing_mappers)
image_mapper_classes: StringRegistry[type] = get_classes(image_preprocessing_mappers)
common_mapper_classes: StringRegistry[type] = get_classes(preprocessing_mappers)

register_mapper_classes(voice_mapper_classes)
register_mapper_classes(text_mapper_classes)
register_mapper_classes(image_mapper_classes)
register_mapper_classes(common_mapper_classes)


def register_default_mapper_config_classes_by_mapper_class(
        mappings: FactoryRegistry[PreprocessingMapperConfig]) -> None:
    default_mapper_config_classes_by_mapper_class.update(mappings)


register_default_mapper_config_classes_by_mapper_class(
    {c: VoicePreprocessingMapperConfig for c in voice_mapper_classes.keys()})
register_default_mapper_config_classes_by_mapper_class(
    {c: TextPreprocessingMapperConfig for c in text_mapper_classes.keys()})
register_default_mapper_config_classes_by_mapper_class(
    {c: ImagePreprocessingMapperConfig for c in image_mapper_classes.keys()})
