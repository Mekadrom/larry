import dataclasses
import functools
from pathlib import Path

import git


@dataclasses.dataclass
class RepoState:
    git_commit: str
    git_dirty: bool


@functools.cache
def git_info() -> RepoState | None:
    try:
        repo = git.Repo(Path(__file__).resolve().parent, search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return None
    return RepoState(repo.head.commit.hexsha, repo.is_dirty(untracked_files=True))
