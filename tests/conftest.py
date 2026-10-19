from collections.abc import Iterator

import pytest

from tests.helpers.common import reset_accelerate_state


@pytest.fixture
def fresh_accelerate_state() -> Iterator[None]:
    reset_accelerate_state()
    yield
    reset_accelerate_state()
