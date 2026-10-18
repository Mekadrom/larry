from typing import Any

import pytest

from larry.common.config.data.data_configs import DatasetMixtureConfig, DatasetSourceConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.voice.config.data.voice_data_configs import ASRMixtureConfig, ASRSourceConfig
from larry.voice.config.training.voice_trainer_configs import SpeechEncoderTrainerConfig

RAW = {
    "train_sources": [
        {"name": "vctk", "index_path": "v", "weight": 0.2, "speaker_namespace": "vctk"},
        {"name": "lj", "index_path": "l", "speaker_id_column": None},
    ],
    "eval_sources": [{"name": "vctk", "index_path": "v", "split": "validation"}],
    "max_batch_seconds": 900.0,
    "pool_size": 30000,
}


def _tokenizer_config(dataset_config: dict[str, Any]) -> SpeechEncoderTrainerConfig:
    return SpeechEncoderTrainerConfig(
        max_steps=1, tokenizer_model_path="t", speaker_mapping_path="s", dataset_config=dataset_config
    )


def test_base_mixture_builds_plain_source_configs() -> None:
    mixture = DatasetMixtureConfig.from_dict({
        "train_sources": [{"name": "a", "index_path": "a"}],
        "eval_sources": [{"name": "b", "index_path": "b"}],
    })

    assert type(mixture) is DatasetMixtureConfig
    assert [type(s) for s in mixture.train_sources + mixture.eval_sources] == [DatasetSourceConfig] * 2


def test_base_mixture_rejects_speech_only_source_fields() -> None:
    with pytest.raises(TypeError, match="audio_column"):
        DatasetMixtureConfig.from_dict({"train_sources": [{"name": "a", "index_path": "a", "audio_column": "x"}]})


def test_speech_mixture_builds_speech_source_configs_with_their_fields() -> None:
    mixture = ASRMixtureConfig.from_dict({**RAW, "sample_rate": 24000})

    assert type(mixture) is ASRMixtureConfig
    assert all(type(s) is ASRSourceConfig for s in mixture.train_sources + mixture.eval_sources)
    vctk, lj = mixture.train_sources
    assert (vctk.name, vctk.weight, vctk.speaker_namespace) == ("vctk", 0.2, "vctk")
    assert lj.speaker_id_column is None
    assert mixture.eval_sources[0].split == "validation"


def test_non_source_keys_reach_the_mixture() -> None:
    mixture = ASRMixtureConfig.from_dict({**RAW, "sample_rate": 24000})

    assert (mixture.max_batch_seconds, mixture.pool_size, mixture.sample_rate) == (900.0, 30000, 24000)


def test_missing_source_lists_default_to_empty() -> None:
    mixture = ASRMixtureConfig.from_dict({"max_batch_seconds": 10.0})

    assert mixture.train_sources == [] and mixture.eval_sources == []


def test_from_dict_does_not_mutate_its_input() -> None:
    raw = {**RAW}

    ASRMixtureConfig.from_dict(raw)

    assert raw == RAW


def test_speech_trainer_config_parses_a_speech_mixture() -> None:
    mixture = _tokenizer_config(RAW).mixture_config

    assert type(mixture) is ASRMixtureConfig
    assert all(type(s) is ASRSourceConfig for s in mixture.train_sources)


def test_mixture_config_is_cached() -> None:
    config = _tokenizer_config(RAW)

    assert config.mixture_config is config.mixture_config


def test_trainer_config_types_name_their_mixture() -> None:
    assert LarryTrainerConfig.mixture_config_type() is DatasetMixtureConfig
    assert SpeechEncoderTrainerConfig.mixture_config_type() is ASRMixtureConfig
    assert ASRMixtureConfig.source_config_type() is ASRSourceConfig
    assert DatasetMixtureConfig.source_config_type() is DatasetSourceConfig
