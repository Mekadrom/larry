from pathlib import Path

import pytest

from larry.common.config.data.preprocessing.dataset_configs import CachedDatasetConfig
from larry.common.data.downloaders import Downloader
from larry.common.data.extractors import Extractor
from tests.helpers.config_recipes import PREPROCESSING_RECIPES, load_pipeline

CACHED = [
    name for name in PREPROCESSING_RECIPES
    if isinstance(load_pipeline(name).dataset, CachedDatasetConfig)
]


@pytest.mark.parametrize("name", CACHED)
def test_every_downloader_and_extractor_key_resolves_to_an_implementation(name: str, tmp_path: Path) -> None:
    dataset = load_pipeline(name).dataset
    assert isinstance(dataset, CachedDatasetConfig)

    for key, config in dataset.downloader_configs():
        Downloader.REGISTRY.resolve(key)(config, tmp_path)
    for key, config in dataset.extractor_configs():
        Extractor.REGISTRY.resolve(key)(config, tmp_path)
