import pytest


@pytest.fixture(autouse=True)
def _fresh_accelerate_state_for_every_test(fresh_accelerate_state: None) -> None:
    ...
