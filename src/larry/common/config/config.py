from pathlib import Path
from typing import Any, Self

from mashumaro.config import BaseConfig
from mashumaro.mixins.yaml import DataClassYAMLMixin
from mashumaro.types import Discriminator


class Config(DataClassYAMLMixin):
    class Config(BaseConfig):
        forbid_extra_keys = True

    @classmethod
    def from_yaml_file(cls, yaml_file_path: Path) -> Self:
        return cls.from_yaml(yaml_file_path.read_bytes())


class TaggedConfig(Config):
    class Config(Config.Config):
        discriminator = Discriminator(
            field="type",
            include_subtypes=True,
            include_supertypes=True,
            variant_tagger_fn=lambda cls: cls.__name__,
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if TaggedConfig in cls.__bases__:
            cls.Config = type("Config", (TaggedConfig.Config,), {})
        super().__init_subclass__(**kwargs)

    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        return {"type": type(self).__name__, **d}
