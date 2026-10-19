import pytest

from tests.helpers.duckdb_checks import zero_rows_test
from tests.helpers.scotus import (
    SEGMENT_ZERO_ROWS,
    segment_connection,
    test_adv_with_justice_surname as test_adv_with_justice_surname,
    test_audio_decodes as test_audio_decodes,
    test_rates as test_rates,
)


@pytest.fixture(scope="session")
def con():
    return segment_connection("SCOTUS_ALIGNED")


test_zero_rows = zero_rows_test(SEGMENT_ZERO_ROWS)
