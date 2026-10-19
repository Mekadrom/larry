import dataclasses
from abc import ABC, abstractmethod

import pytest

from larry.common.config.config import TaggedConfig
from larry.common.utils.registrable import Registrable


@dataclasses.dataclass(kw_only=True)
class ShapeConfig(TaggedConfig):
    sides: int = 0


@dataclasses.dataclass(kw_only=True)
class SquareConfig(ShapeConfig):
    sides: int = 4


@dataclasses.dataclass(kw_only=True)
class RedSquareConfig(SquareConfig):
    color: str = "red"


@dataclasses.dataclass(kw_only=True)
class TriangleConfig(ShapeConfig):
    sides: int = 3


class Shape[C: ShapeConfig](Registrable, ABC, root=True, config_root=ShapeConfig):
    @abstractmethod
    def draw(self) -> str:
        ...


class Square(Shape[SquareConfig]):
    def draw(self) -> str:
        return "square"


class Triangle(Shape[TriangleConfig]):
    def draw(self) -> str:
        return "triangle"


class AlsoTriangle(Shape[TriangleConfig]):
    def draw(self) -> str:
        return "triangle"


class Helper(Shape[SquareConfig], register=False):
    def draw(self) -> str:
        return "helper"


def test_an_implementation_is_found_from_the_config_type() -> None:
    assert Shape.REGISTRY.for_config(SquareConfig()) is Square


def test_a_config_subclass_falls_back_to_its_nearest_declared_ancestor() -> None:
    assert Shape.REGISTRY.for_config(RedSquareConfig()) is Square


def test_two_implementations_declaring_one_config_type_is_an_error() -> None:
    with pytest.raises(TypeError, match="AlsoTriangle"):
        Shape.REGISTRY.for_config(TriangleConfig())


def test_a_config_without_an_implementation_is_reported() -> None:
    with pytest.raises(KeyError, match="ShapeConfig"):
        Shape.REGISTRY.for_config(ShapeConfig())


def test_the_type_tag_selects_the_subclass_and_is_rendered_back() -> None:
    config = ShapeConfig.from_dict({"type": "RedSquareConfig", "color": "blue"})

    assert config == RedSquareConfig(color="blue")
    assert config.to_dict() == {"type": "RedSquareConfig", "sides": 4, "color": "blue"}


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(Exception, match="colour"):
        ShapeConfig.from_dict({"type": "RedSquareConfig", "colour": "blue"})
