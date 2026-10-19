from larry.image.config.preprocessing.image_dataset_configs import GoogleDOCCIMediaCachedDatasetConfig
from larry.voice.config.preprocessing.voice_dataset_configs import (
    CommonVoiceDatasetConfig,
    VCTKCachedDatasetProviderConfig,
)


def test_media_dir_follows_an_overridden_archive_root() -> None:
    assert CommonVoiceDatasetConfig(download_extract_cache_dir="c").media_dir == "cv-corpus-27.0-2026-09-11/en"
    assert CommonVoiceDatasetConfig(download_extract_cache_dir="c", archive_root="cv-28/en").media_dir == "cv-28/en"
    assert VCTKCachedDatasetProviderConfig(download_extract_cache_dir="c", archive_root="wav").media_dir == "wav"
    docci = GoogleDOCCIMediaCachedDatasetConfig(path="p", download_extract_cache_dir="c", archive_root="i")
    assert docci.media_dir == "i"


def test_an_explicit_media_dir_wins_over_archive_root() -> None:
    config = CommonVoiceDatasetConfig(download_extract_cache_dir="c", archive_root="r", media_dir="m")

    assert config.media_dir == "m"


def test_vctk_download_path_follows_an_overridden_archive_file() -> None:
    assert VCTKCachedDatasetProviderConfig(download_extract_cache_dir="c").download_path == "download/DS_10283_3443.zip"
    assert VCTKCachedDatasetProviderConfig(download_extract_cache_dir="c", archive_file="a.zip").download_path == (
        "download/a.zip"
    )
