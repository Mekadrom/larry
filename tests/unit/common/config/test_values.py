import argparse
from pathlib import Path

import pytest
import yaml

from larry.common.config.values import ConfigValues, RunRecord, WorkingTree
from larry.common.utils import git_utils
from larry.common.utils.git_utils import RepoState


def _write(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_maps_deep_merge_and_later_files_win(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", {"trainer": {"lr": 1.0, "seed": 1, "nested": {"a": 1, "b": 2}}})
    overlay = _write(tmp_path / "overlay.yaml", {"trainer": {"lr": 2.0, "nested": {"b": 3, "c": 4}}})

    merged = ConfigValues([base, overlay]).merged()

    assert merged == {"trainer": {"lr": 2.0, "seed": 1, "nested": {"a": 1, "b": 3, "c": 4}}}


def test_lists_are_replaced_and_null_sets_none(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", {"names": ["a", "b"], "column": "speaker_id"})
    overlay = _write(tmp_path / "overlay.yaml", {"names": ["c"], "column": None})

    assert ConfigValues([base, overlay]).merged() == {"names": ["c"], "column": None}


def test_delete_removes_a_key_from_what_came_before(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", {"model": {"n_masks": 0, "d_model": 8}})
    overlay = _write(tmp_path / "overlay.yaml", {"model": {"n_masks": {"$delete": True}}})

    assert ConfigValues([base, overlay]).merged() == {"model": {"d_model": 8}}


def test_new_entries_append_unless_positioned(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", {"pipeline": {"a": {"x": 1}, "b": {"x": 2}}})
    overlay = _write(tmp_path / "overlay.yaml", {"pipeline": {
        "c": {"x": 3},
        "first": {"$before": "a", "x": 0},
        "middle": {"$after": "a", "x": 9},
    }})

    merged = ConfigValues([base, overlay]).merged()

    assert list(merged["pipeline"]) == ["first", "a", "middle", "b", "c"]
    assert merged["pipeline"]["middle"] == {"x": 9}


def test_an_existing_entry_can_be_moved(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", {"pipeline": {"a": {"x": 1}, "b": {"x": 2}}})
    overlay = _write(tmp_path / "overlay.yaml", {"pipeline": {"b": {"$before": "a"}}})

    merged = ConfigValues([base, overlay]).merged()

    assert list(merged["pipeline"]) == ["b", "a"]
    assert merged["pipeline"]["b"] == {"x": 2}


@pytest.mark.parametrize("entry", [
    {"$after": "missing"},
    {"$before": "a", "$after": "a"},
    {"$delete": True, "$after": "a"},
    {"$unknown": 1},
])
def test_invalid_directives_are_rejected(tmp_path: Path, entry: dict[str, object]) -> None:
    base = _write(tmp_path / "base.yaml", {"pipeline": {"a": {"x": 1}}})
    overlay = _write(tmp_path / "overlay.yaml", {"pipeline": {"new": entry}})

    with pytest.raises(ValueError):
        ConfigValues([base, overlay]).merged()


def test_sets_apply_after_every_file_and_parse_their_values_as_yaml(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", {"trainer": {"lr": 1.0, "betas": [0.9, 0.99]}})

    merged = ConfigValues([base], ["trainer.lr=5.0e-4", "trainer.betas=[0.8, 0.9]", "trainer.name=~"]).merged()

    assert merged == {"trainer": {"lr": 5e-4, "betas": [0.8, 0.9], "name": None}}


@pytest.mark.parametrize("assignment", ["trainer.lr", "=1"])
def test_malformed_sets_are_rejected(assignment: str) -> None:
    with pytest.raises(ValueError):
        ConfigValues.parse_set(assignment)


def test_cli_arguments_collect_files_sets_and_the_dirty_flag() -> None:
    parser = argparse.ArgumentParser()
    ConfigValues.add_arguments(parser)

    args = parser.parse_args(["-f", "a.yaml", "--values", "b.yaml", "--set", "x=1", "--allow_dirty"])
    values = ConfigValues.from_args(args)

    assert [f.name for f in values.files] == ["a.yaml", "b.yaml"]
    assert values.sets == ["x=1"]
    assert args.allow_dirty


@pytest.mark.parametrize(("state", "allow_dirty", "blocked"), [
    (RepoState("abc", git_dirty=True), False, True),
    (RepoState("abc", git_dirty=True), True, False),
    (RepoState("abc", git_dirty=False), False, False),
    (None, False, False),
])
def test_a_dirty_tree_blocks_unless_allowed(
        monkeypatch: pytest.MonkeyPatch, state: RepoState | None, allow_dirty: bool, blocked: bool
) -> None:
    monkeypatch.setattr(git_utils, "git_info", lambda: state)

    if blocked:
        with pytest.raises(SystemExit, match="--allow_dirty"):
            WorkingTree.ensure_clean(allow_dirty)
    else:
        WorkingTree.ensure_clean(allow_dirty)


def test_a_run_record_writes_the_rendered_config_and_cli(tmp_path: Path) -> None:
    record = RunRecord(config={"trainer": {"type": "X", "lr": 0.5}}, cli="training -f a.yaml")

    record.write(tmp_path / "run")

    assert yaml.safe_load((tmp_path / "run" / RunRecord.CONFIG_FILE).read_text()) == record.config
    assert (tmp_path / "run" / RunRecord.CLI_FILE).read_text() == "training -f a.yaml\n"
