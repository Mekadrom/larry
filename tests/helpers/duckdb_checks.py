import os
from collections.abc import Callable

import duckdb
import pytest


def env_connection(env_var: str) -> tuple[duckdb.DuckDBPyConnection, str]:
    root = os.environ.get(env_var)
    if root is None:
        pytest.skip(f"{env_var} not set")
    return duckdb.connect(), root


def zero_rows_test(queries: dict[str, str]) -> Callable[..., None]:
    @pytest.mark.parametrize("sql", queries.values(), ids=queries.keys())
    def test_zero_rows(con, sql):
        rows = con.execute(sql).fetchall()
        assert not rows, f"{len(rows)} rows, e.g. {rows[:10]}"

    return test_zero_rows
