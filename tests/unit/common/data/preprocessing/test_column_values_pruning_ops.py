from collections import Counter
from typing import Any

import pytest
from datasets import Dataset

from larry.common.config.data.preprocessing.pipeline_config import Provenance, ProvenanceAncestry
from larry.common.config.data.preprocessing.pruner_configs import ColumnValuesPrunerConfig
from larry.common.data.preprocessing.pruners import ColumnValuesPruner


@pytest.mark.parametrize(
    ("op_config", "example"),
    [
        ({"containing": True, "case_sensitive": False, "values": ["SA", "ND", "NC"]}, "CC-BY-SA"),
        ({"containing": True, "case_sensitive": False, "values": ["SA", "ND", "NC"]}, "CC-BY-ND"),
        ({"containing": True, "case_sensitive": False, "values": ["SA", "ND", "NC"]}, "CC-BY-NC"),
        ({"containing": True, "case_sensitive": False, "values": ["SA", "ND", "NC"]}, "cc-by-sa"),
        ({"containing": True, "case_sensitive": False, "values": ["sa", "nd", "nc"]}, "CC-BY-ND"),
        ({"case_sensitive": False, "values": ["success", "partial_success"]}, "SUCCESS"),
        ({"containing": True, "case_sensitive": False, "values": ["SUCCESS"]}, "partial_success"),
        ({"values": ["success", "partial_success"]}, "success"),
        ({"values": [None, ""]}, ""),
        ({"values": [None, ""]}, "   \t\n"),
    ]
)
def test_op_in_returns_true(op_config: dict[str, Any], example: str) -> None:
    assert ColumnValuesPruner.op_matches(op_config, example)


@pytest.mark.parametrize(
    ("op_config", "example"),
    [
        ({"containing": True, "case_sensitive": False, "values": ["SA", "ND", "NC"]}, "CC-BY-4.0"),
        ({"containing": True, "case_sensitive": False, "values": ["SA", "ND", "NC"]}, "PD"),
        ({"containing": True, "case_sensitive": False, "values": ["SA", "ND", "NC"]}, "CC0"),
        ({"containing": True, "values": ["SA", "ND", "NC"]}, "cc-by-sa"),
        ({"containing": True, "values": ["PARTIAL_SUCCESS"]}, "SUCCESS"),
        ({"values": ["PARTIAL_SUCCESS"]}, "SUCCESS"),
        ({"values": ["SA", "ND", "NC"]}, "sa"),
        ({"values": [None, ""]}, None),
    ]
)
def test_op_in_returns_false(op_config: dict[str, Any], example: str) -> None:
    assert not ColumnValuesPruner.op_matches(op_config, example)


def test_column_values_pruning_status() -> None:
    config = ColumnValuesPrunerConfig(
        input_columns=["status"],
        op="matches",
        op_config={
            "containing": True,
            "case_sensitive": False,
            "values": ["success"],
        },
        cache_results=False,
        num_proc=1,
    )

    sut = ColumnValuesPruner(Provenance(
        id_columns=[],
        git_commit="",
        git_is_dirty=None,
        produced_on="bruh",
        ancestry=ProvenanceAncestry(
            datasets=[],
            models=[],
        ),
    ), config)

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
    config = ColumnValuesPrunerConfig(
        op="nmatches",
        input_columns=["exif_copyright"],
        op_config={
            "containing": True,
            "case_sensitive": False,
            "values": ["SA", "NC", "ND"],
        },
        cache_results=False,
        num_proc=1,
    )

    sut = ColumnValuesPruner(Provenance(
        id_columns=[],
        git_commit="",
        git_is_dirty=None,
        produced_on="bruh",
        ancestry=ProvenanceAncestry(
            datasets=[],
            models=[],
        ),
    ), config)

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
