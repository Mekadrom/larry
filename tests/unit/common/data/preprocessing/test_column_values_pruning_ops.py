from collections import Counter
from typing import Any

import pytest
from datasets import Dataset

from larry.common.config.data.preprocessing.pruner_configs import ColumnValuesPrunerConfig, MatchOpConfig
from larry.common.data.preprocessing.pruners import ColumnValuesPruner
from tests.helpers.common import provenance

LICENSE_TERMS = {"containing": True, "case_sensitive": False, "values": ["SA", "ND", "NC"]}
BLANK = {"values": [None, ""]}


@pytest.mark.parametrize(
    ("op_config", "example"),
    [
        (LICENSE_TERMS, "CC-BY-SA"),
        (LICENSE_TERMS, "CC-BY-ND"),
        (LICENSE_TERMS, "CC-BY-NC"),
        (LICENSE_TERMS, "cc-by-sa"),
        ({"containing": True, "case_sensitive": False, "values": ["sa", "nd", "nc"]}, "CC-BY-ND"),
        ({"case_sensitive": False, "values": ["success", "partial_success"]}, "SUCCESS"),
        ({"containing": True, "case_sensitive": False, "values": ["SUCCESS"]}, "partial_success"),
        ({"values": ["success", "partial_success"]}, "success"),
        (BLANK, ""),
        (BLANK, "   \t\n"),
    ]
)
def test_op_in_returns_true(op_config: dict[str, Any], example: str) -> None:
    assert ColumnValuesPruner.op_matches(MatchOpConfig(**op_config), example)


@pytest.mark.parametrize(
    ("op_config", "example"),
    [
        (LICENSE_TERMS, "CC-BY-4.0"),
        (LICENSE_TERMS, "PD"),
        (LICENSE_TERMS, "CC0"),
        ({"containing": True, "values": ["SA", "ND", "NC"]}, "cc-by-sa"),
        ({"containing": True, "values": ["PARTIAL_SUCCESS"]}, "SUCCESS"),
        ({"values": ["PARTIAL_SUCCESS"]}, "SUCCESS"),
        ({"values": ["SA", "ND", "NC"]}, "sa"),
        (BLANK, None),
    ]
)
def test_op_in_returns_false(op_config: dict[str, Any], example: str) -> None:
    assert not ColumnValuesPruner.op_matches(MatchOpConfig(**op_config), example)


def _pruned(column: str, values: list[str], op: str, op_config: dict[str, Any]) -> Counter[str]:
    config = ColumnValuesPrunerConfig(
        input_columns=[column], op=op, op_config=MatchOpConfig(**op_config), cache_results=False, num_proc=1
    )
    dataset = ColumnValuesPruner(provenance(), config).preprocess_dataset(Dataset.from_dict({column: values}))
    return Counter(dataset[column])


def test_column_values_pruning_status() -> None:
    kept = _pruned(
        "status",
        ["success", "SUCCESS", "partial_success", "PARTIAL_SUCCESS", "error"],
        op="matches",
        op_config={"containing": True, "case_sensitive": False, "values": ["success"]},
    )

    assert kept == Counter(["success", "SUCCESS", "partial_success", "PARTIAL_SUCCESS"])


def test_column_values_pruning() -> None:
    kept = _pruned(
        "exif_copyright",
        ["cc-by-sa", "CC-BY-SA", "cc-by-nc", "CC-BY-NC", "cc-by-nd", "CC-BY-ND", "cc0", "CC0", "pd", "PD"],
        op="nmatches",
        op_config={"containing": True, "case_sensitive": False, "values": ["SA", "NC", "ND"]},
    )

    assert kept == Counter(["cc0", "CC0", "pd", "PD"])
