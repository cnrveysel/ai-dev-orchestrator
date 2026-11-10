"""Read-mostly git integration.

Attribution strategy: before and after each worker attempt the full working
tree is snapshotted into a tree object using a TEMPORARY index file
(GIT_INDEX_FILE). The real index, working tree and refs are never touched.
The worker's changes are exactly diff(pre_tree, post_tree), so pre-existing
user modifications are never attributed to the worker.

Generated/dependency paths (node_modules/, dist/, build/, ... see path_filter) are
excluded from snapshots, attribution and diffs regardless of .gitignore; they are
only counted. Lockfiles are listed but kept out of the stored diff.

This module never runs destructive commands (no reset/clean/push/rebase). The only
writes to the working tree are the V2 repair restorations (restore_tree / write_blob):
they write back file contents recorded in an earlier snapshot, through a TEMPORARY index,
so the real index, HEAD and refs are never touched, generated/ignored paths are never
touched, and the replaced state is itself kept as a snapshot tree by the caller.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.integrations import path_filter
from app.models.attempt import ChangedFile
from app.workers.process_runner import ProcessRunner, ProcessResult

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

LogFn = Callable[[str], None]


class GitError(RuntimeError):
    pass


@dataclass
class StatusEntry:
    xy: str
    path: str
    orig_path: str | None = None


@dataclass
class RepoInfo:
    is_repo: bool
    toplevel: str | None = None
    branch: str | None = None
    head: str | None = None
    dirty: list[StatusEntry] = field(default_factory=list)
    error: str = ""

    @property
    def is_dirty(self) -> bool:
        return bool(self.dirty)

    def describe(self) -> str:
        if not self.is_repo:
            return "not a git repository" + (f" ({self.error})" if self.error else "")
        head = (self.head or "no commits")[:10]
        state = f"{len(self.dirty)} changed/untracked" if self.dirty else "clean"
        return f"{self.branch or '(detached)'} @ {head} - {state}"


@dataclass
class Snapshot:
    tree: str | None  # None when precise snapshots are unavailable
    status: dict[str, str]  # authored path -> XY (generated paths excluded)
    hashes: dict[str, str]  # fallback content hashes of dirty files
    generated: dict[str, str] = field(default_factory=dict)  # generated path -> XY (only counted)


def snapshot_to_dict(s: Snapshot) -> dict:
    """Persistable form (V2 phase/run base). Generated paths are only counted, never stored."""
    return {"tree": s.tree, "status": dict(s.status), "hashes": dict(s.hashes)}


def snapshot_from_dict(d: dict, generated: dict[str, str] | None = None) -> Snapshot:
    return Snapshot(d.get("tree") or None, dict(d.get("status") or {}), dict(d.get("hashes") or {}),
                    dict(generated or {}))


@dataclass
class AttemptChanges:
    changed_files: list[ChangedFile]
    preexisting_dirty: list[str]
    diff_stat: str
    diff: str
    diff_truncated: bool = False
    generated_excluded: int = 0  # generated paths changed by the attempt (not attributed)
    generated_by_dir: dict[str, int] = field(default_factory=dict)


def parse_porcelain_z(data: str) -> list[StatusEntry]:
    entries: list[StatusEntry] = []
    parts = data.split("\0")
    i = 0
    while i < len(parts):
        item = parts[i]
        i += 1
        if len(item) < 4:
            continue
        xy, path = item[:2], item[3:]
        orig = None
        if xy[0] in "RC" or xy[1] in "RC":
            orig = parts[i] if i < len(parts) else None
            i += 1
        entries.append(StatusEntry(xy, path, orig))
    return entries


def parse_name_status_z(data: str) -> list[tuple[str, str, str | None]]:
    out: list[tuple[str, str, str | None]] = []
    parts = [p for p in data.split("\0")]
    i = 0
    while i < len(parts):
        code = parts[i]
        i += 1
        if not code:
            continue
        if code[0] in "RC":
            old, new = parts[i], parts[i + 1]
            i += 2
            out.append((code[0], new, old))
        else:
            out.append((code[0], parts[i], None))
            i += 1
    return out


class GitRepo:
    def __init__(self, path: str, log: LogFn | None = None, max_diff_chars: int = 200_000):
        self.path = path
        self.log = log or (lambda s: None)
        self.max_diff_chars = max_diff_chars
        self._git_exe = shutil.which("git") or "git"

    async def _git(self, *args: str, env: dict[str, str] | None = None,
                   check: bool = True, timeout: float = 120, stdin: str | None = None) -> ProcessResult:
        argv = [self._git_exe, "-c", "core.quotepath=off", "-c", "color.ui=false", *args]
        res = await ProcessRunner().run(argv, cwd=self.path, env_overrides=env, timeout=timeout, stdin_data=stdin)
        if check and res.exit_code != 0:
            raise GitError(f"git {' '.join(args)} failed ({res.exit_code}): {res.stderr.strip()[:500]}")
        return res

    async def info(self) -> RepoInfo:
        if not os.path.isdir(self.path):
            return RepoInfo(False, error="directory does not exist")
        try:
            r = await self._git("rev-parse", "--show-toplevel", check=False)
        except Exception as exc:
            return RepoInfo(False, error=str(exc))
        if r.exit_code != 0:
            return RepoInfo(False)
        top = r.stdout.strip()
        # Guard: the selected folder must be the repository root; otherwise we
        # would snapshot/diff a parent repository (e.g. a git repo at %USERPROFILE%).
        if os.path.normcase(os.path.abspath(top)) != os.path.normcase(os.path.abspath(self.path)):
            return RepoInfo(False, toplevel=top,
                            error=f"folder is inside a parent repository at {top}, not a repository root")
        br = await self._git("symbolic-ref", "--short", "-q", "HEAD", check=False)
        head = await self._git("rev-parse", "--verify", "-q", "HEAD", check=False)
        st = await self.status()
        return RepoInfo(True, toplevel=top,
                        branch=br.stdout.strip() or None,
                        head=head.stdout.strip() or None,
                        dirty=st)

    async def status(self) -> list[StatusEntry]:
        r = await self._git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        return parse_porcelain_z(r.stdout)

    async def head(self) -> str | None:
        r = await self._git("rev-parse", "--verify", "-q", "HEAD", check=False)
        return r.stdout.strip() or None

    async def create_branch(self, name: str) -> None:
        """Non-destructive: `git switch -c` carries uncommitted changes along."""
        await self._git("check-ref-format", "--branch", name)
        await self._git("switch", "-c", name)

    # ------------------------------------------------------------ snapshots

    async def snapshot(self, precise: bool = True) -> Snapshot:
        entries = await self.status()
        status: dict[str, str] = {}
        generated: dict[str, str] = {}
        for e in entries:
            (generated if path_filter.is_generated(e.path) else status)[e.path] = e.xy
        tree = None
        if precise:
            try:
                tree = await self._snapshot_tree()
            except Exception as exc:
                self.log(f"precise snapshot unavailable, falling back to status hashes: {exc}")
        hashes = {} if tree else await self._hash_paths([p for p, xy in status.items() if "D" not in xy])
        return Snapshot(tree, status, hashes, generated)

    async def _snapshot_tree(self) -> str:
        r = await self._git("rev-parse", "--git-path", "index")
        real_index = Path(r.stdout.strip())
        if not real_index.is_absolute():
            real_index = Path(self.path) / real_index
        fd, tmp = tempfile.mkstemp(prefix="aio-index-", suffix=".tmp")
        os.close(fd)
        try:
            if real_index.is_file():
                shutil.copyfile(real_index, tmp)  # reuse stat cache: only changed files get rehashed
            else:
                os.unlink(tmp)
            env = {"GIT_INDEX_FILE": tmp}
            # Generated paths are never added: no hashing of node_modules etc. into .git/objects.
            await self._git("add", "-A", "--", ".", *path_filter.git_exclude_pathspecs(), env=env, timeout=600)
            w = await self._git("write-tree", env=env)
            return w.stdout.strip()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    async def _hash_paths(self, paths: list[str]) -> dict[str, str]:
        files = [p for p in paths if os.path.isfile(os.path.join(self.path, p))]
        if not files:
            return {}
        argv = [self._git_exe, "hash-object", "--stdin-paths"]
        res = await ProcessRunner().run(argv, cwd=self.path, stdin_data="\n".join(files) + "\n", timeout=300)
        hashes = res.stdout.split()
        return dict(zip(files, hashes)) if len(hashes) == len(files) else {}

    async def tree_diff_stat(self, old_tree: str, new_tree: str) -> str:
        """`git diff --stat` between two snapshot trees, generated paths excluded."""
        r = await self._git("diff", "--stat=160", old_tree, new_tree, "--", ".", *path_filter.git_exclude_pathspecs())
        return r.stdout.strip()

    async def base_tree(self) -> str:
        head = await self.head()
        return f"{head}^{{tree}}" if head else EMPTY_TREE

    async def attempt_changes(self, pre: Snapshot, post: Snapshot) -> AttemptChanges:
        preexisting = sorted(pre.status)
        excl = path_filter.git_exclude_pathspecs()
        if pre.tree and post.tree:
            ns = await self._git("diff", "--no-renames", "--name-status", "-z", pre.tree, post.tree, "--", ".", *excl)
            changed = []
            for code, path, old in parse_name_status_z(ns.stdout):
                note = "file also had pre-existing user changes" if path in pre.status else ""
                changed.append(ChangedFile(path, code, True, note))
            stat = (await self._git("diff", "--stat=160", pre.tree, post.tree, "--", ".", *excl)).stdout
            diff = (await self._git("diff", pre.tree, post.tree, "--", ".", *excl,
                                    *path_filter.git_lockfile_exclude_pathspecs())).stdout
        else:
            changed = []
            all_paths = set(pre.status) | set(post.status)
            for path in sorted(all_paths):
                before, after = pre.status.get(path), post.status.get(path)
                if before == after and pre.hashes.get(path) == post.hashes.get(path):
                    continue
                code = "D" if after and "D" in after else ("A" if before is None and after and "?" in after else "M")
                if after is None:
                    code = "M"  # file returned to HEAD state
                note = "file also had pre-existing user changes (diff may include them)" if before else ""
                changed.append(ChangedFile(path, code, True, note))
            clean_before = [c.path for c in changed
                            if c.path not in pre.status and path_filter.classify(c.path) == path_filter.DIFFABLE]
            if clean_before:
                stat = (await self._git("diff", "--stat=160", "HEAD", "--", *clean_before, check=False)).stdout
                diff = (await self._git("diff", "HEAD", "--", *clean_before, check=False)).stdout
            else:
                stat, diff = "", ""
        # Defensive: never attribute generated paths, even if a pathspec missed one.
        changed = [c for c in changed if not path_filter.is_generated(c.path)]
        gen_changed = sorted(p for p in set(pre.generated) | set(post.generated)
                             if pre.generated.get(p) != post.generated.get(p))
        truncated = len(diff) > self.max_diff_chars
        if truncated:
            diff = diff[: self.max_diff_chars] + "\n...[diff truncated]..."
        return AttemptChanges(changed, preexisting, stat.strip(), diff, truncated,
                              generated_excluded=len(gen_changed),
                              generated_by_dir=dict(path_filter.summarize_generated(gen_changed)))

    # ------------------------------------------------------------ V2 repair: restore / exact blobs

    async def tree_changes(self, old_tree: str, new_tree: str) -> list[tuple[str, str]]:
        """(code, path) for authored paths that differ between two snapshot trees
        (A = only in new_tree, D = only in old_tree, M/T = content/type changed)."""
        ns = await self._git("diff", "--no-renames", "--name-status", "-z", old_tree, new_tree, "--", ".",
                             *path_filter.git_exclude_pathspecs())
        return [(code, path) for code, path, _old in parse_name_status_z(ns.stdout)]

    async def restore_tree(self, target_tree: str, current_tree: str,
                           exact: dict[str, str] | None = None) -> list[tuple[str, str]]:
        """Make the authored working-tree files equal to `target_tree` (a snapshot of this
        working tree): files that differ are rewritten from it, files created after it are
        removed. Only paths that differ between the two snapshots are touched; generated and
        ignored paths are not part of the snapshots and are never touched. `exact` maps paths to
        --no-filters blobs recorded with the target snapshot: those paths are written back byte
        for byte (a tree holds EOL-normalized content, so checkout could change line endings).
        Returns (code, path) of what was restored ("W" rewritten, "D" removed)."""
        exact = exact or {}
        changes = await self.tree_changes(current_tree, target_tree)
        writes = [p for code, p in changes if code != "D"]
        deletes = [p for code, p in changes if code == "D"]
        exact_writes = [p for p in writes if p in exact]
        tree_writes = [p for p in writes if p not in exact]
        for p in exact_writes:
            await self.write_blob(p, exact[p])
        if tree_writes:
            fd, tmp = tempfile.mkstemp(prefix="aio-restore-", suffix=".tmp")
            os.close(fd)
            os.unlink(tmp)  # git creates the temporary index itself
            env = {"GIT_INDEX_FILE": tmp}
            try:
                await self._git("read-tree", target_tree, env=env)
                await self._git("checkout-index", "-f", "-z", "--stdin", env=env,
                                stdin="\0".join(tree_writes) + "\0", timeout=600)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        root = os.path.abspath(self.path)
        for p in deletes:
            full = os.path.abspath(os.path.join(root, p))
            if full.startswith(root + os.sep) and os.path.isfile(full):
                os.remove(full)
        return [("W", p) for p in writes] + [("D", p) for p in deletes]

    async def hash_files(self, paths: list[str], write: bool = False) -> dict[str, str]:
        """Exact-byte blob ids (--no-filters) of existing files; with write=True the blobs are
        stored in the object database so write_blob() can restore them byte for byte."""
        files = [p for p in paths if "\n" not in p and os.path.isfile(os.path.join(self.path, p))]
        if not files:
            return {}
        args = ["hash-object", "--no-filters", *(["-w"] if write else []), "--stdin-paths"]
        res = await self._git(*args, stdin="\n".join(files) + "\n", timeout=300)
        ids = res.stdout.split()
        if len(ids) != len(files):
            raise GitError(f"hash-object returned {len(ids)} ids for {len(files)} files")
        return dict(zip(files, ids))

    async def write_blob(self, path: str, blob: str) -> None:
        """Restore one file byte for byte from a blob stored by hash_files(write=True)."""
        root = os.path.abspath(self.path)
        full = os.path.abspath(os.path.join(root, path))
        if not full.startswith(root + os.sep):
            raise GitError(f"refusing to write outside the repository: {path}")
        data = await self._git_bytes("cat-file", "blob", blob)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        tmp = full + ".aio-restore.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, full)

    async def _git_bytes(self, *args: str, timeout: float = 120) -> bytes:
        """Binary-safe git call (ProcessRunner decodes text)."""
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = await asyncio.create_subprocess_exec(
            self._git_exe, *args, cwd=self.path, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except BaseException:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise
        if proc.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed ({proc.returncode}): "
                           f"{err.decode('utf-8', 'replace').strip()[:500]}")
        return out
