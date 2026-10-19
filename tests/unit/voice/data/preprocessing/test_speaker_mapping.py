import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.voice.data.preprocessing.speaker_mapping import SpeakerMapping, SpeakerMappingCreator
from tests.helpers.common import write_parquet_split


def _index(root: Path, split: str, speaker_ids: list[Any], files: int = 1) -> str:
    return str(write_parquet_split(root, split, {"speaker_id": speaker_ids}, files))


def _create(tmp_path: Path, sources: list[SpeechSourceConfig]) -> SpeakerMapping:
    SpeakerMappingCreator(sources, tmp_path / "out" / "speakers").run()
    return SpeakerMapping.from_csv(tmp_path / "out" / "speakers.csv")


def test_csv_round_trip_keeps_every_key_and_id(tmp_path: Path) -> None:
    mapping = SpeakerMapping()
    for key in ["vctk:p225", "vctk:p226", "lj-speech:0"]:
        mapping.add_item(key)

    mapping.save_csv(tmp_path / "speakers")
    loaded = SpeakerMapping.from_csv(tmp_path / "speakers.csv")

    assert loaded.ids == mapping.ids
    assert loaded.items == mapping.items


def test_add_item_returns_the_existing_id_for_a_known_key() -> None:
    mapping = SpeakerMapping()

    assert [mapping.add_item(k) for k in ["a", "b", "a"]] == [0, 1, 0]
    assert len(mapping) == 2


def test_csv_with_a_gap_in_ids_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "speakers.csv"
    path.write_text("key,id\na,0\nb,2\n")

    with pytest.raises(ValueError, match="dense"):
        SpeakerMapping.from_csv(path)


def test_speakers_are_deduplicated_across_shards_and_splits(tmp_path: Path) -> None:
    root = tmp_path / "vctk"
    _index(root, "train.a", ["p2", "p1", "p2"], files=2)
    _index(root, "train.b", ["p1", "p3"])

    mapping = _create(tmp_path, [SpeechSourceConfig(name="vctk", index_path=str(root), split="train.a+train.b")])

    assert mapping.items == ["vctk:p1", "vctk:p2", "vctk:p3"]


def test_train_and_eval_sources_with_the_same_name_share_ids(tmp_path: Path) -> None:
    root = tmp_path / "vctk"
    _index(root, "train", ["p1", "p2"])
    _index(root, "validation", ["p2", "p9"])

    mapping = _create(tmp_path, [
        SpeechSourceConfig(name="vctk", index_path=str(root), split="train"),
        SpeechSourceConfig(name="vctk", index_path=str(root), split="validation"),
    ])

    assert mapping.items == ["vctk:p1", "vctk:p2", "vctk:p9"]


def test_speaker_namespace_shares_ids_between_differently_named_sources(tmp_path: Path) -> None:
    train = _index(tmp_path / "heavy", "train", ["100", "200"])
    test = _index(tmp_path / "heavy_test", "train", ["200", "300"])

    mapping = _create(tmp_path, [
        SpeechSourceConfig(name="heavy", index_path=train),
        SpeechSourceConfig(name="heavy-test", index_path=test, speaker_namespace="heavy"),
    ])

    assert mapping.items == ["heavy:100", "heavy:200", "heavy:300"]


def test_same_speaker_string_in_different_sources_gets_different_ids(tmp_path: Path) -> None:
    a = _index(tmp_path / "a", "train", ["1"])
    b = _index(tmp_path / "b", "train", ["1"])

    mapping = _create(tmp_path, [
        SpeechSourceConfig(name="a", index_path=a),
        SpeechSourceConfig(name="b", index_path=b),
    ])

    assert mapping["a:1"] != mapping["b:1"]


def test_single_speaker_source_gets_one_id(tmp_path: Path) -> None:
    mapping = _create(tmp_path, [
        SpeechSourceConfig(name="lj-speech", index_path=str(tmp_path / "missing"), speaker_id_column=None),
    ])

    assert len(mapping) == 1
    assert mapping.items[0].startswith("lj-speech:")


def test_integer_speaker_ids_keep_zero_and_skip_missing(tmp_path: Path) -> None:
    root = _index(tmp_path / "ls", "train", [0, 84, None, 84])

    mapping = _create(tmp_path, [SpeechSourceConfig(name="ls", index_path=root)])

    assert mapping.items == ["ls:0", "ls:84"]


def test_ids_do_not_depend_on_string_hash_randomization(tmp_path: Path) -> None:
    root = _index(tmp_path / "src", "train", [f"speaker{i}" for i in range(200)])
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from larry.voice.config.data.voice_data_configs import SpeechSourceConfig\n"
        "from larry.voice.data.preprocessing.speaker_mapping import SpeakerMappingCreator\n"
        f"SpeakerMappingCreator([SpeechSourceConfig(name='s', index_path={root!r})], Path(sys.argv[1])).run()\n"
    )
    outputs = []
    for seed in ("1", "2"):
        prefix = tmp_path / f"seed{seed}"
        subprocess.run([sys.executable, "-c", script, str(prefix)], check=True,
                       env={**os.environ, "PYTHONHASHSEED": seed})
        outputs.append(prefix.with_suffix(".csv").read_text())

    assert outputs[0] == outputs[1]


def test_key_uses_the_namespace_when_set_and_the_name_otherwise() -> None:
    plain = SpeechSourceConfig(name="vctk", index_path="v")
    shared = SpeechSourceConfig(name="heavy-test", index_path="h", speaker_namespace="heavy")

    assert SpeakerMapping.key(plain, "p1") == "vctk:p1"
    assert SpeakerMapping.key(shared, 84) == "heavy:84"


def test_single_speaker_key_ignores_the_speaker_id() -> None:
    source = SpeechSourceConfig(name="lj", index_path="l", speaker_id_column=None)

    assert SpeakerMapping.key(source, None) == SpeakerMapping.key(source, "anything")


def test_created_keys_match_mapping_keys(tmp_path: Path) -> None:
    root = _index(tmp_path / "v", "train", ["p1"])
    sources = [
        SpeechSourceConfig(name="vctk", index_path=root),
        SpeechSourceConfig(name="lj", index_path=str(tmp_path / "missing"), speaker_id_column=None),
    ]

    mapping = _create(tmp_path, sources)

    assert mapping.items == [SpeakerMapping.key(sources[0], "p1"), SpeakerMapping.key(sources[1], None)]


def test_read_source_takes_speakers_from_the_source_by_row(tmp_path: Path) -> None:
    write_parquet_split(tmp_path / "materialized", "train", {"id": ["a", "b", "c"], "speaker_id": ["s1", "s2", "s3"]})
    write_parquet_split(tmp_path / "index", "train", {"row": [0, 2], "id": ["a", "c"]})
    source = SpeechSourceConfig(name="m", index_path=str(tmp_path / "index"), source_path=str(tmp_path / "materialized"))

    assert SpeakerMappingCreator.read_source(source) == ["s1", "s3"]
