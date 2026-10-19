import pytest
from mashumaro.exceptions import ExtraKeysError, InvalidFieldValue

from larry.common.config.data.data_configs import DatasetMixtureConfig, DatasetSourceConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.voice.config.data.voice_data_configs import ASRMixtureConfig, ASRSourceConfig
from larry.voice.config.training.voice_trainer_configs import SpeechEncoderTrainerConfig

RAW = {
    "train_sources": {
        "vctk": {"index_path": "v", "weight": 0.2, "speaker_namespace": "vctk"},
        "lj": {"index_path": "l", "speaker_id_column": None},
    },
    "eval_sources": {"vctk": {"index_path": "v", "split": "validation"}},
    "max_batch_seconds": 900.0,
    "pool_size": 30000,
}


def _tokenizer_config(mixture_config: dict[str, object]) -> LarryTrainerConfig:
    return LarryTrainerConfig.from_dict({
        "type": "SpeechEncoderTrainerConfig", "max_steps": 1, "tokenizer_model_path": "t",
        "speaker_mapping_path": "s", "mixture_config": mixture_config,
    })


def test_base_mixture_builds_plain_source_configs() -> None:
    mixture = DatasetMixtureConfig.from_dict({
        "train_sources": {"a": {"index_path": "a"}},
        "eval_sources": {"b": {"index_path": "b"}},
    })

    assert type(mixture) is DatasetMixtureConfig
    assert [type(s) for s in mixture.train_sources + mixture.eval_sources] == [DatasetSourceConfig] * 2


def test_base_mixture_rejects_speech_only_source_fields() -> None:
    with pytest.raises((ExtraKeysError, InvalidFieldValue)):
        DatasetMixtureConfig.from_dict({"train_sources": {"a": {"index_path": "a", "audio_column": "x"}}})


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
    config = _tokenizer_config(RAW)

    assert type(config) is SpeechEncoderTrainerConfig
    assert type(config.mixture_config) is ASRMixtureConfig
    assert all(type(s) is ASRSourceConfig for s in config.mixture_config.train_sources)


def test_source_entries_take_their_name_from_their_key_and_render_back_as_a_map() -> None:
    mixture = ASRMixtureConfig.from_dict(RAW)

    assert [s.name for s in mixture.train_sources] == ["vctk", "lj"]
    assert list(mixture.to_dict()["train_sources"]) == ["vctk", "lj"]
    assert "name" not in mixture.to_dict()["train_sources"]["vctk"]
    assert ASRMixtureConfig.from_dict(mixture.to_dict()) == mixture


def test_a_source_name_that_contradicts_its_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        ASRMixtureConfig.from_dict({"train_sources": {"vctk": {"name": "lj", "index_path": "v"}}})
