import asyncio
import subprocess
from pathlib import Path

import pytest

from app.integrations.git import GitRepo


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("one\n")
    (r / "b.txt").write_text("bee\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    return r


def index_bytes(repo: Path) -> bytes:
    return (repo / ".git" / "index").read_bytes()


@pytest.mark.parametrize("precise", [True, False])
def test_attempt_attribution_excludes_preexisting_changes(repo, precise):
    async def main():
        g = GitRepo(str(repo))
        # pre-existing user work
        (repo / "a.txt").write_text("user edit\n")
        (repo / "user_untracked.txt").write_text("mine\n")
        idx_before = index_bytes(repo)
        pre = await g.snapshot(precise)
        # "worker" changes
        (repo / "b.txt").write_text("worker edit\n")
        (repo / "new_file.py").write_text("print(1)\n")
        post = await g.snapshot(precise)
        ch = await g.attempt_changes(pre, post)
        assert index_bytes(repo) == idx_before, "real index must not be modified"
        return ch

    ch = asyncio.run(main())
    worker = {c.path for c in ch.changed_files}
    assert worker == {"b.txt", "new_file.py"}
    assert "a.txt" not in worker and "user_untracked.txt" not in worker
    assert set(ch.preexisting_dirty) == {"a.txt", "user_untracked.txt"}
    assert "worker edit" in ch.diff
    assert "user edit" not in ch.diff


def test_precise_diff_of_file_with_preexisting_changes(repo):
    async def main():
        g = GitRepo(str(repo))
        (repo / "a.txt").write_text("user edit\n")
        pre = await g.snapshot(True)
        (repo / "a.txt").write_text("user edit\nworker line\n")
        post = await g.snapshot(True)
        return await g.attempt_changes(pre, post)

    ch = asyncio.run(main())
    assert [c.path for c in ch.changed_files] == ["a.txt"]
    assert ch.changed_files[0].note
    assert "+worker line" in ch.diff
    assert "-one" not in ch.diff  # the user's change is not part of the attempt diff


def test_info_and_parent_repo_guard(repo):
    info = asyncio.run(GitRepo(str(repo)).info())
    assert info.is_repo and info.branch == "main" and info.head
    sub = repo / "sub"
    sub.mkdir()
    info2 = asyncio.run(GitRepo(str(sub)).info())
    assert not info2.is_repo and "parent repository" in info2.error


def test_create_branch_keeps_uncommitted_changes(repo):
    (repo / "a.txt").write_text("dirty\n")
    asyncio.run(GitRepo(str(repo)).create_branch("ai-orchestrator/run-1"))
    out = subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True).stdout
    assert out.strip() == "ai-orchestrator/run-1"
    assert (repo / "a.txt").read_text() == "dirty\n"
