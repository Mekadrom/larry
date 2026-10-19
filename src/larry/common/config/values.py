import argparse
import copy
import dataclasses
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from larry.common.config.config import Config


class ConfigValues:
    DELETE = "$delete"
    BEFORE = "$before"
    AFTER = "$after"
    DIRECTIVES = (DELETE, BEFORE, AFTER)

    def __init__(self, files: Sequence[Path], sets: Sequence[str] = ()) -> None:
        self.files = [Path(f).expanduser().resolve() for f in files]
        self.sets = list(sets)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-f",
            "--values",
            dest="values",
            type=Path,
            action="append",
            required=True,
            help="A values yaml file; repeat to overlay files, later files winning.",
        )
        parser.add_argument(
            "--set",
            dest="sets",
            action="append",
            default=[],
            help="A dotted override like trainer.lr=5e-4, applied after every values file.",
        )
        parser.add_argument(
            "--allow_dirty",
            action="store_true",
            help="Runs even if the git working tree has uncommitted or untracked changes.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ConfigValues:
        return cls(args.values, args.sets)

    def merged(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for file in self.files:
            layer = yaml.safe_load(file.read_text()) or {}
            if not isinstance(layer, Mapping):
                raise TypeError(f"{file} must contain a mapping at the top level")

            merged = self.merge(merged, layer)

        for assignment in self.sets:
            merged = self.merge(merged, self.parse_set(assignment))

        return self.strip_directives(merged)

    def load[C: Config](self, definition_type: type[C]) -> C:
        return definition_type.from_dict(self.merged())

    @staticmethod
    def parse_set(assignment: str) -> dict[str, Any]:
        path, separator, raw = assignment.partition("=")
        if not separator or not path:
            raise ValueError(f"--set expects key.path=value, got {assignment!r}")

        value: Any = yaml.safe_load(raw)
        for key in reversed(path.split(".")):
            value = {key: value}

        return value

    @classmethod
    def merge(cls, base: Any, overlay: Any) -> Any:
        if not isinstance(base, Mapping) or not isinstance(overlay, Mapping):
            return copy.deepcopy(overlay)

        merged = dict(base)
        for key, value in overlay.items():
            if isinstance(value, Mapping) and value.get(cls.DELETE):
                cls._check_directives(key, value, allowed=(cls.DELETE,))
                merged.pop(key, None)
                continue

            if isinstance(value, Mapping):
                cls._check_directives(key, value, allowed=(cls.BEFORE, cls.AFTER))
                merged[key] = cls.merge(merged.get(key), value)
            else:
                merged[key] = copy.deepcopy(value)

            if isinstance(value, Mapping) and (cls.BEFORE in value or cls.AFTER in value):
                merged = cls._position(merged, key, value)

        return merged

    @classmethod
    def _check_directives(cls, key: str, value: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
        directives = [k for k in value if isinstance(k, str) and k.startswith("$")]
        unknown = [d for d in directives if d not in cls.DIRECTIVES]
        if unknown:
            raise ValueError(f"{key!r} has unknown directives {unknown}; known: {list(cls.DIRECTIVES)}")

        misplaced = [d for d in directives if d not in allowed]
        if misplaced:
            raise ValueError(f"{key!r} combines {misplaced} with {[d for d in directives if d in allowed]}")

        if cls.BEFORE in value and cls.AFTER in value:
            raise ValueError(f"{key!r} sets both {cls.BEFORE} and {cls.AFTER}")

    @classmethod
    def _position(cls, merged: dict[str, Any], key: str, value: Mapping[str, Any]) -> dict[str, Any]:
        anchor = value.get(cls.BEFORE, value.get(cls.AFTER))
        if anchor not in merged or anchor == key:
            raise ValueError(f"{key!r} is positioned relative to {anchor!r}, which is not a sibling key")

        entry = merged.pop(key)
        reordered: dict[str, Any] = {}
        for k, v in merged.items():
            if k == anchor and cls.BEFORE in value:
                reordered[key] = entry

            reordered[k] = v
            if k == anchor and cls.AFTER in value:
                reordered[key] = entry

        return reordered

    @classmethod
    def strip_directives(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: cls.strip_directives(v) for k, v in value.items() if k not in cls.DIRECTIVES}

        if isinstance(value, list):
            return [cls.strip_directives(v) for v in value]

        return value


@dataclasses.dataclass(kw_only=True)
class RunRecord:
    config: dict[str, Any]
    cli: str

    CONFIG_FILE = "config.resolved.yaml"
    CLI_FILE = "cli.txt"

    @classmethod
    def of(cls, definition: Config) -> RunRecord:
        return cls(config=definition.to_dict(), cli=shlex.join(sys.orig_argv))

    def config_yaml(self) -> str:
        return yaml.safe_dump(self.config, sort_keys=False)

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.CONFIG_FILE).write_text(self.config_yaml())
        (directory / self.CLI_FILE).write_text(self.cli + "\n")
