from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig, SingleColumnMapperConfig, \
    ValueOverrideMapperConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance, ProvenanceAncestry
from larry.common.data.preprocessing.mappers import Mapper, SingleColumnMapper, ValueOverrideMapper


def _provenance() -> Provenance:
    return Provenance(
        id_columns=[],
        produced_on="",
        git_commit="",
        git_is_dirty=None,
        ancestry=ProvenanceAncestry(datasets=set(), models=set()),
    )


class LengthMapper(SingleColumnMapper[str | None, int | None, SingleColumnMapperConfig]):
    def preprocess_example(self, example: str | None) -> int | None:
        if example is None or example == "drop":
            return None
        return len(example)

    def output_feature(self) -> Value:
        return Value("int64")


class UpperMapper(SingleColumnMapper[str, str, SingleColumnMapperConfig]):
    def preprocess_example(self, example: str) -> str:
        return example.upper()

    def output_feature(self) -> Value:
        return Value("string")


class PairMapper(Mapper[MapperConfig]):
    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        a, b = self.config.input_columns
        total, label = self.config.output_columns
        return {
            total: [None if x is None or y is None else x + y for x, y in zip(batch[a], batch[b])],
            label: [None if x is None else f"{x}" for x in batch[a]],
        }

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        total, label = self.config.output_columns
        return Features({total: Value("int64"), label: Value("string")})


def _base(**kwargs: Any) -> dict[str, Any]:
    return {"num_proc": 1, "batch_size": 2, "writer_batch_size": 2, "cache_results": False, **kwargs}


def _dataset() -> Dataset:
    return Dataset.from_dict({
        "audio": [b"a", b"b", b"c", b"d", b"e"],
        "text": ["x", "yy", "zzz", "wwww", "vvvvv"],
        "speaker": ["s0", None, "s2", "s3", "s4"],
    })


def test_single_column_mapper_keeps_every_other_column_and_appends_output() -> None:
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(**_base(input_column="text", output_column="n")))

    out = mapper.preprocess_dataset(_dataset())

    assert out.column_names == ["audio", "text", "speaker", "n"]
    assert out["n"] == [1, 2, 3, 4, 5]
    assert out["audio"] == _dataset()["audio"]
    assert out["speaker"] == _dataset()["speaker"]


def test_input_column_is_kept_unless_configured_for_removal() -> None:
    kept = LengthMapper(_provenance(), SingleColumnMapperConfig(**_base(input_column="text", output_column="n")))
    removed = LengthMapper(_provenance(), SingleColumnMapperConfig(
        **_base(input_column="text", output_column="n", remove_columns=["text"])
    ))

    assert "text" in kept.preprocess_dataset(_dataset()).column_names
    assert "text" not in removed.preprocess_dataset(_dataset()).column_names


def test_remove_columns_can_name_columns_the_mapper_does_not_read() -> None:
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(
        **_base(input_column="text", output_column="n", remove_columns=["audio"])
    ))

    out = mapper.preprocess_dataset(_dataset())

    assert out.column_names == ["text", "speaker", "n"]


def test_in_place_output_replaces_the_input_column() -> None:
    mapper = UpperMapper(_provenance(), SingleColumnMapperConfig(**_base(input_column="text", output_column="text")))

    out = mapper.preprocess_dataset(_dataset())

    assert sorted(out.column_names) == ["audio", "speaker", "text"]
    assert out["text"] == ["X", "YY", "ZZZ", "WWWW", "VVVVV"]


def test_existing_output_column_is_overwritten_not_duplicated() -> None:
    dataset = _dataset().add_column("n", [0] * 5)
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(**_base(input_column="text", output_column="n")))

    out = mapper.preprocess_dataset(dataset)

    assert out.column_names.count("n") == 1
    assert out["n"] == [1, 2, 3, 4, 5]


def test_source_with_indices_mapping_stays_row_aligned() -> None:
    dataset = _dataset().filter(lambda row: row["text"] != "yy")
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(**_base(input_column="text", output_column="n")))

    out = mapper.preprocess_dataset(dataset)

    assert out["text"] == ["x", "zzz", "wwww", "vvvvv"]
    assert out["n"] == [1, 3, 4, 5]


def test_dataset_dict_splits_are_mapped_independently() -> None:
    dataset = DatasetDict({"train": _dataset(), "validation": _dataset().select(range(2))})
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(**_base(input_column="text", output_column="n")))

    out = mapper.preprocess_dataset(dataset)

    assert {k: v["n"] for k, v in out.items()} == {"train": [1, 2, 3, 4, 5], "validation": [1, 2]}


def test_multiple_processes_produce_the_same_result() -> None:
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(
        **_base(input_column="text", output_column="n", num_proc=2)
    ))

    out = mapper.preprocess_dataset(_dataset())

    assert out["n"] == [1, 2, 3, 4, 5]


def test_filter_null_outputs_drops_rows_with_null_outputs_only() -> None:
    dataset = _dataset().map(lambda row: {"text": "drop" if row["text"] == "zzz" else row["text"]})
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(
        **_base(input_column="text", output_column="n", filter_null_outputs=True)
    ))

    out = mapper.preprocess_dataset(dataset)

    assert out["text"] == ["x", "yy", "wwww", "vvvvv"]
    assert out["speaker"] == ["s0", None, "s3", "s4"]


def test_null_in_unrelated_column_does_not_trigger_filtering() -> None:
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(
        **_base(input_column="text", output_column="n", filter_null_outputs=True)
    ))

    out = mapper.preprocess_dataset(_dataset())

    assert len(out) == 5


def test_multi_column_mapper_reads_and_writes_configured_columns() -> None:
    dataset = Dataset.from_dict({"a": [1, 2, None], "b": [10, 20, 30], "audio": [b"x", b"y", b"z"]})
    mapper = PairMapper(_provenance(), MapperConfig(
        **_base(input_columns=["a", "b"], output_columns=["total", "label"])
    ))

    out = mapper.preprocess_dataset(dataset)

    assert sorted(out.column_names) == sorted(["a", "b", "audio", "total", "label"])
    assert out["total"] == [11, 22, None]
    assert out["label"] == ["1", "2", None]


def test_multi_column_mapper_filters_rows_with_a_null_in_any_output() -> None:
    dataset = Dataset.from_dict({"a": [1, 2, None], "b": [10, None, 30]})
    mapper = PairMapper(_provenance(), MapperConfig(
        **_base(input_columns=["a", "b"], output_columns=["total", "label"], filter_null_outputs=True)
    ))

    out = mapper.preprocess_dataset(dataset)

    assert out["a"] == [1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_column": "text", "output_column": "n"},
        {"input_column": "text", "output_column": "n", "input_columns": ["text"], "output_columns": ["n"]},
    ],
)
def test_single_column_config_exposes_column_lists(kwargs: dict[str, Any]) -> None:
    config = SingleColumnMapperConfig(**_base(**kwargs))

    assert config.input_columns == ["text"]
    assert config.output_columns == ["n"]


def test_single_column_config_rejects_conflicting_column_lists() -> None:
    with pytest.raises(ValueError):
        SingleColumnMapperConfig(**_base(input_column="text", output_column="n", input_columns=["audio"]))


def test_value_override_mapper_applies_rules_sharing_an_input_column() -> None:
    dataset = Dataset.from_dict({
        "docket": ["13-7120rearg", "1-1"],
        "pdf_url": ["old", "keep"],
        "date_argued": ["2014-11-05", "2020-01-01"],
        "term": ["2014", "2019"],
    })
    config = ValueOverrideMapperConfig(**_base(mappings_config=[
        {"input_column": "docket", "input_column_value": "13-7120rearg",
         "output_column": "pdf_url", "output_column_value": "new"},
        {"input_column": "docket", "input_column_value": "13-7120rearg",
         "output_column": "date_argued", "output_column_value": "2015-04-20"},
    ]))

    out = ValueOverrideMapper(_provenance(), config).preprocess_dataset(dataset)

    assert sorted(out.column_names) == sorted(["date_argued", "docket", "pdf_url", "term"])
    assert out["pdf_url"] == ["new", "keep"]
    assert out["date_argued"] == ["2015-04-20", "2020-01-01"]
    assert out["docket"] == ["13-7120rearg", "1-1"]
    assert out["term"] == ["2014", "2019"]


def test_columns_removed_from_a_filtered_source_are_not_rewritten(tmp_path: Path) -> None:
    _dataset().save_to_disk(tmp_path / "source")
    dataset = load_from_disk(tmp_path / "source").filter(lambda row: row["text"] != "yy")
    mapper = LengthMapper(_provenance(), SingleColumnMapperConfig(**_base(
        input_column="text", output_column="n", remove_columns=["audio"], num_proc=2, cache_results=True,
    )))

    out = mapper.preprocess_dataset(dataset)

    assert out["text"] == ["x", "zzz", "wwww", "vvvvv"]
    assert out["n"] == [1, 3, 4, 5]
    for cache_file in out.cache_files:
        with pa.memory_map(cache_file["filename"]) as source:
            assert "audio" not in pa.ipc.open_stream(source).schema.names


def test_chained_mappers_on_disk_backed_data_survive_multiprocess_sharding(tmp_path: Path) -> None:
    _dataset().save_to_disk(tmp_path / "source")
    dataset = load_from_disk(tmp_path / "source")

    def config(input_column: str, output_column: str, num_proc: int) -> SingleColumnMapperConfig:
        return SingleColumnMapperConfig(**_base(
            input_column=input_column, output_column=output_column, num_proc=num_proc, cache_results=True,
        ))

    dataset = UpperMapper(_provenance(), config("text", "upper", 1)).preprocess_dataset(dataset)
    dataset = LengthMapper(_provenance(), config("upper", "n", 2)).preprocess_dataset(dataset)

    assert dataset.column_names == ["audio", "text", "speaker", "upper", "n"]
    assert dataset["n"] == [1, 2, 3, 4, 5]


def test_rerun_with_a_fresh_provenance_reuses_the_cached_result(tmp_path: Path) -> None:
    _dataset().save_to_disk(tmp_path / "source")
    config = SingleColumnMapperConfig(**_base(input_column="text", output_column="n", cache_results=True))

    ancestry = ProvenanceAncestry.from_dict({})
    first = LengthMapper(Provenance.new_provenance({}, ancestry), config).preprocess_dataset(
        load_from_disk(tmp_path / "source"))
    second = LengthMapper(Provenance.new_provenance({}, ancestry), config).preprocess_dataset(
        load_from_disk(tmp_path / "source"))

    assert second._fingerprint == first._fingerprint


def test_fresh_provenance_has_no_per_run_fields() -> None:
    provenance = Provenance.new_provenance({}, ProvenanceAncestry.from_dict({}))

    assert (provenance.produced_on, provenance.git_commit, provenance.git_is_dirty) == (None, None, None)
