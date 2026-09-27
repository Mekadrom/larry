from collections import Counter
from typing import Any

import pytest
from datasets import Dataset

from larry.common.config.preprocessing_mapper_configs import ColumnValuesPruningPreprocessingMapperConfig
from larry.common.data.preprocessing.preprocessing_mappers import ColumnValuesPruningPreprocessingMapper


@pytest.mark.parametrize(
    ("op_config", "values", "example"),
    [
        ({"containing": True, "case_sensitive": False}, ["SA", "ND", "NC"], "CC-BY-SA"),
        ({"containing": True, "case_sensitive": False}, ["SA", "ND", "NC"], "CC-BY-ND"),
        ({"containing": True, "case_sensitive": False}, ["SA", "ND", "NC"], "CC-BY-NC"),
        ({"containing": True, "case_sensitive": False}, ["SA", "ND", "NC"], "cc-by-sa"),
        ({"containing": True, "case_sensitive": False}, ["sa", "nd", "nc"], "CC-BY-ND"),
        ({"case_sensitive": False}, ["success", "partial_success"], "SUCCESS"),
        ({"containing": True, "case_sensitive": False}, ["SUCCESS"], "partial_success"),
        ({}, ["success", "partial_success"], "success"),
        ({}, [None, ""], None),
        ({}, [None, ""], ""),
        ({}, [None, ""], "   \t\n"),
    ]
)
def test_op_in_returns_true(op_config: dict[str, Any], values: list[Any | None], example: str) -> None:
    assert ColumnValuesPruningPreprocessingMapper.op_in(op_config, values, example)


@pytest.mark.parametrize(
    ("op_config", "values", "example"),
    [
        ({"containing": True, "case_sensitive": False}, ["SA", "ND", "NC"], "CC-BY-4.0"),
        ({"containing": True, "case_sensitive": False}, ["SA", "ND", "NC"], "PD"),
        ({"containing": True, "case_sensitive": False}, ["SA", "ND", "NC"], "CC0"),
        ({"containing": True}, ["SA", "ND", "NC"], "cc-by-sa"),
        ({"containing": True}, ["PARTIAL_SUCCESS"], "SUCCESS"),
        ({}, ["PARTIAL_SUCCESS"], "SUCCESS"),
        ({}, ["SA", "ND", "NC"], "sa"),
    ]
)
def test_op_in_returns_false(op_config: dict[str, Any], values: list[Any | None], example: str) -> None:
    assert not ColumnValuesPruningPreprocessingMapper.op_in(op_config, values, example)


def test_column_values_pruning_status() -> None:
    config = ColumnValuesPruningPreprocessingMapperConfig(
        values=["success"],
        op="nin",
        input_column="status",
        op_config={
            "containing": True,
            "case_sensitive": False
        }
    )

    sut = ColumnValuesPruningPreprocessingMapper(config)

    dataset = Dataset.from_dict(
        {
            "status": [
                "success",
                "SUCCESS",
                "partial_success",
                "PARTIAL_SUCCESS",
                "error",
            ]
        }
    )

    dataset = sut.preprocess_dataset(dataset)

    assert Counter(dataset["status"]) == Counter([
        "success",
        "SUCCESS",
        "partial_success",
        "PARTIAL_SUCCESS",
    ])


def test_column_values_pruning() -> None:
    config = ColumnValuesPruningPreprocessingMapperConfig(
        values=["SA", "NC", "ND"],
        op="in",
        input_column="exif_copyright",
        op_config={
            "containing": True,
            "case_sensitive": False
        }
    )

    sut = ColumnValuesPruningPreprocessingMapper(config)

    dataset = Dataset.from_dict(
        {
            "exif_copyright": [
                "cc-by-sa",
                "CC-BY-SA",
                "cc-by-nc",
                "CC-BY-NC",
                "cc-by-nd",
                "CC-BY-ND",
                "cc0",
                "CC0",
                "pd",
                "PD",
            ]
        }
    )

    dataset = sut.preprocess_dataset(dataset)

    assert Counter(dataset["exif_copyright"]) == Counter(["cc0", "CC0", "pd", "PD"])
