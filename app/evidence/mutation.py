"""Repository-mutation protection for evidence collection (verification must never change the project).

    git       the working tree is snapshotted (tree + exact bytes of changed/untracked files) before the
              evidence steps run and restored exactly afterwards
    manifest  a directory that is not a git repository: a BOUNDED manifest (path, size, mtime, content
              hash) of the authored files. Within the byte budget the monitored files are also copied to
              a snapshot OUTSIDE the project, so changes can be restored exactly; beyond it changes are
              only DETECTED and reported (restoration cannot be guaranteed); beyond the file budget only
              the first files are monitored (partial)

Generated/dependency directories (the same list the git snapshots exclude) and .git are never
scanned. Whatever protection was actually active is reported with the evidence - protection is
never claimed when it was not available.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

from app.integrations import path_filter

HASH_LIMIT = 2 * 1024 * 1024  # files larger than this are compared by size + mtime only
MAX_HASH_TOTAL = 128 * 1024 * 1024
_EXCLUDED = frozenset({".git"} | {d.lower() for d in path_filter.GENERATED_DIRS})


@dataclass
class ManifestScan:
    entries: dict[str, tuple[int, int, str]] = field(default_factory=dict)  # rel -> (size, mtime_ns, sha1 | "")
    truncated: bool = False  # the file budget was reached: only the first files are monitored
    total_bytes: int = 0

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for rel in sorted(self.entries):
            size, mtime, sha = self.entries[rel]
            h.update(f"{rel}\0{size}\0{sha or mtime}\n".encode("utf-8", errors="replace"))
        return h.hexdigest()[:16]


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_scan(root: str, *, max_files: int = 5000, hash_limit: int = HASH_LIMIT) -> ManifestScan:
    """Bounded, read-only scan of the authored files under root (symlinks are never followed)."""
    scan = ManifestScan()
    hashed = 0
    root = os.path.abspath(root)
    for dirpath, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in _EXCLUDED
                         and not os.path.islink(os.path.join(dirpath, d)))
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if os.path.islink(full) or path_filter.is_generated(rel):
                continue
            if len(scan.entries) >= max_files:
                scan.truncated = True
                return scan
            try:
                st = os.stat(full)
                sha = ""
                if st.st_size <= hash_limit and hashed + st.st_size <= MAX_HASH_TOTAL:
                    sha = _sha1_file(full)
                    hashed += st.st_size
            except OSError:
                continue
            scan.entries[rel] = (st.st_size, st.st_mtime_ns, sha)
            scan.total_bytes += st.st_size
    return scan


def diff_scans(before: ManifestScan, after: ManifestScan) -> tuple[list[str], list[str], list[str]]:
    """(changed, added, deleted) relative paths."""
    changed = sorted(p for p in set(before.entries) & set(after.entries) if before.entries[p] != after.entries[p]
                     and not _same_content(before.entries[p], after.entries[p]))
    added = sorted(set(after.entries) - set(before.entries))
    deleted = sorted(set(before.entries) - set(after.entries))
    if before.truncated or after.truncated:  # beyond the budget nothing is known: never report it as added/deleted
        known = set(before.entries) & set(after.entries)
        added = [p for p in added if p in known]
        deleted = [p for p in deleted if p in known]
    return changed, added, deleted


def _same_content(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    # Same size and the same content hash: an mtime touch alone is not a change.
    return a[0] == b[0] and bool(a[2]) and a[2] == b[2]


@dataclass
class MutationReport:
    changed: list[str] = field(default_factory=list)  # every path the evidence steps changed, added or removed
    restored: list[str] = field(default_factory=list)
    unrestored: list[str] = field(default_factory=list)  # changes that could not be restored
    detail: str = ""

    @property
    def detected(self) -> bool:
        return bool(self.changed)


class MutationGuard:
    kind = "none"

    async def before(self) -> Any:
        return None

    async def after(self, token: Any) -> MutationReport:
        return MutationReport()

    def describe(self, token: Any) -> dict[str, Any]:
        return {"kind": "none", "exact": False, "detail": "no repository-mutation protection"}


class GitMutationGuard(MutationGuard):
    """V4 pass 2 behaviour: snapshot tree + exact blobs before, exact restoration after."""
    kind = "git"

    def __init__(self, git: Any, log: Callable[[str], None] = lambda s: None, max_exact: int = 500):
        self.git, self.log, self.max_exact = git, log, max_exact

    async def before(self) -> Any:
        try:
            snap = await self.git.snapshot(True)
            if not snap.tree:
                self.log("repository guard unavailable (no precise git snapshot); evidence steps run unguarded")
                return None
            paths = [p for p, xy in sorted(snap.status.items()) if "D" not in xy][:self.max_exact]
            return snap.tree, await self.git.hash_files(paths, write=True)
        except Exception as exc:
            self.log(f"repository guard unavailable ({exc}); evidence steps run unguarded")
            return None

    async def after(self, token: Any) -> MutationReport:
        if token is None:
            return MutationReport(detail="unguarded")
        tree, exact = token
        try:
            cur = await self.git.snapshot(True)
            if not cur.tree or cur.tree == tree:
                return MutationReport()
            paths = [p for _, p in await self.git.restore_tree(tree, cur.tree, exact=exact)]
            if paths:
                self.log(f"evidence steps modified {len(paths)} repository file(s); restored: {', '.join(paths[:8])}")
            return MutationReport(changed=paths, restored=list(paths), detail="restored from the git snapshot")
        except Exception as exc:
            self.log(f"repository guard could not compare/restore ({exc})")
            return MutationReport(detail=f"could not compare/restore: {exc}"[:300])

    def describe(self, token: Any) -> dict[str, Any]:
        if token is None:
            return {"kind": "none", "exact": False, "detail": "git snapshot unavailable: evidence ran unguarded"}
        return {"kind": "git", "exact": True, "detail": "git snapshot (exact restoration)"}


class ManifestMutationGuard(MutationGuard):
    """Non-git directories: bounded manifest; exact restoration only when the snapshot fits the budget."""
    kind = "manifest"

    def __init__(self, repo_path: str, snapshot_root: str = "", *, max_files: int = 5000,
                 max_snapshot_bytes: int = 64 * 1024 * 1024, log: Callable[[str], None] = lambda s: None):
        self.repo = os.path.abspath(repo_path)
        self.snapshot_root = snapshot_root
        self.max_files, self.max_snapshot_bytes, self.log = max(1, max_files), max(0, max_snapshot_bytes), log

    async def before(self) -> Any:
        return await asyncio.to_thread(self._before)

    def _before(self) -> dict[str, Any]:
        scan = manifest_scan(self.repo, max_files=self.max_files)
        token: dict[str, Any] = {"scan": scan, "dir": "", "exact": False}
        if scan.truncated or scan.total_bytes > self.max_snapshot_bytes:
            return token  # detection only (or partial): restoration cannot be guaranteed
        base = self.snapshot_root or tempfile.gettempdir()
        try:
            os.makedirs(base, exist_ok=True)
            snap_dir = tempfile.mkdtemp(prefix="guard-", dir=base)
            for rel in scan.entries:
                dst = os.path.join(snap_dir, *rel.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(self.repo, *rel.split("/")), dst)
            token.update(dir=snap_dir, exact=True)
        except OSError as exc:
            self.log(f"repository guard: exact snapshot unavailable ({exc}); changes are only detected")
        return token

    async def after(self, token: Any) -> MutationReport:
        if not token:
            return MutationReport(detail="unguarded")
        try:
            return await asyncio.to_thread(self._after, token)
        finally:
            if token.get("dir"):
                shutil.rmtree(token["dir"], ignore_errors=True)

    def _inside(self, rel: str) -> str:
        full = os.path.abspath(os.path.join(self.repo, *rel.split("/")))
        return full if full.startswith(self.repo + os.sep) else ""

    def _after(self, token: dict[str, Any]) -> MutationReport:
        before: ManifestScan = token["scan"]
        after = manifest_scan(self.repo, max_files=self.max_files)
        changed, added, deleted = diff_scans(before, after)
        touched = sorted(set(changed) | set(added) | set(deleted))
        if not touched:
            return MutationReport()
        if not token.get("exact"):
            self.log(f"evidence steps changed {len(touched)} file(s) of a non-git directory; restoration cannot be "
                     "guaranteed (no exact snapshot within the budget): " + ", ".join(touched[:8]))
            return MutationReport(changed=touched, unrestored=touched,
                                  detail="detected by the bounded manifest; no exact snapshot, NOT restored")
        restored, failed = [], []
        for rel in changed + deleted:
            src, dst = os.path.join(token["dir"], *rel.split("/")), self._inside(rel)
            try:
                if not dst:
                    raise OSError("outside the directory")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = dst + ".aio-restore.tmp"
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                restored.append(rel)
            except OSError:
                failed.append(rel)
        for rel in added:  # files the evidence steps created: removed again (regular files only)
            full = self._inside(rel)
            try:
                if full and os.path.isfile(full) and not os.path.islink(full):
                    os.remove(full)
                    restored.append(rel)
                else:
                    failed.append(rel)
            except OSError:
                failed.append(rel)
        self.log(f"evidence steps changed {len(touched)} file(s); restored {len(restored)} from the bounded snapshot"
                 + (f", could NOT restore: {', '.join(failed[:6])}" if failed else ""))
        return MutationReport(changed=touched, restored=sorted(restored), unrestored=sorted(failed),
                              detail="restored from the bounded snapshot" if not failed else
                              "partly restored from the bounded snapshot")

    def describe(self, token: Any) -> dict[str, Any]:
        if not token:
            return {"kind": "none", "exact": False, "detail": "no protection"}
        scan: ManifestScan = token["scan"]
        if scan.truncated:
            detail = (f"bounded manifest of the first {len(scan.entries)} files only (file budget reached): partial "
                      "detection, no restoration")
        elif token.get("exact"):
            detail = f"bounded manifest of {len(scan.entries)} files with an exact snapshot (restorable)"
        else:
            detail = (f"bounded manifest of {len(scan.entries)} files, detection only ({scan.total_bytes} bytes exceed "
                      "the snapshot budget: restoration cannot be guaranteed)")
        return {"kind": "manifest", "exact": bool(token.get("exact")), "files": len(scan.entries),
                "partial": scan.truncated, "detail": detail}


async def repo_fingerprint(repo_path: str, git: Any = None, *, max_files: int = 5000) -> tuple[str, str]:
    """(fingerprint, kind) of the authored state of the project: git_tree | git_status | manifest, or ("", "")
    when it cannot be established (e.g. a directory beyond the manifest budget)."""
    if git is not None:
        try:
            snap = await git.snapshot(True)
        except Exception:
            return "", ""
        if snap.tree:
            return f"git:{snap.tree}", "git_tree"
        data = repr((sorted(snap.status.items()), sorted(snap.hashes.items())))
        return "gs:" + hashlib.sha1(data.encode("utf-8", errors="replace")).hexdigest()[:16], "git_status"
    if not repo_path or not os.path.isdir(repo_path):
        return "", ""
    scan = await asyncio.to_thread(manifest_scan, repo_path, max_files=max_files)
    if scan.truncated:
        return "", ""
    return "mf:" + scan.fingerprint(), "manifest"
