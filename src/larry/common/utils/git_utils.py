import dataclasses
import functools
from pathlib import Path

import git


@dataclasses.dataclass
class RepoState:
    """Helper dataclass for storing git information."""

    git_commit: str
    git_dirty: bool


@functools.cache
def git_info() -> RepoState | None:
    """Returns git info for storing in provenance."""
    try:
        repo = git.Repo(Path(__file__).resolve().parent, search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return None
    
    return RepoState(repo.head.commit.hexsha, repo.is_dirty(untracked_files=True))


class WorkingTree:
    @staticmethod
    def ensure_clean(allow_dirty: bool) -> None:
        repo_state = git_info()
        if repo_state is None or not repo_state.git_dirty or allow_dirty:
            return

        raise SystemExit(
            f"the git working tree at {repo_state.git_commit[:12]} has uncommitted or untracked changes; "
            "commit them for a reproducible run or pass --allow_dirty"
        )