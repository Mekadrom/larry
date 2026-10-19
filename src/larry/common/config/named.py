from collections.abc import Iterable, Mapping
from typing import Any


class NamedEntries:
    KEY = "name"

    @staticmethod
    def to_lists(d: dict[Any, Any], fields: Iterable[str]) -> dict[Any, Any]:
        d = dict(d)
        for field in fields:
            entries = d.get(field)
            if isinstance(entries, Mapping):
                d[field] = [NamedEntries._named(key, value) for key, value in entries.items()]

        return d

    @staticmethod
    def to_maps(d: dict[Any, Any], fields: Iterable[str]) -> dict[Any, Any]:
        d = dict(d)
        for field in fields:
            entries = d.get(field)
            if isinstance(entries, list):
                d[field] = {
                    entry[NamedEntries.KEY]: {k: v for k, v in entry.items() if k != NamedEntries.KEY}
                    for entry in entries
                }
        return d

    @staticmethod
    def _named(key: str, value: Any) -> dict[Any, Any]:
        if value is None:
            value = {}

        if not isinstance(value, Mapping):
            raise TypeError(f"entry {key!r} must be a mapping, got {type(value).__name__}")

        if NamedEntries.KEY in value and value[NamedEntries.KEY] != key:
            raise ValueError(f"entry {key!r} also sets {NamedEntries.KEY}={value[NamedEntries.KEY]!r}")

        return {NamedEntries.KEY: key, **value}
