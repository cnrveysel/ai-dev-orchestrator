"""Evidence artifacts: bounded files OUTSIDE the target repository (AI Studio's data folder), with
only their metadata (id, kind, media type, reference, size) persisted in the run state.

Artifacts are generic: logs and reports (text), and binary artifacts such as captured images
(media_type image/png with width/height/capture metadata). An image artifact records WHAT was
captured - it is never evidence of what the image shows.

Storage is rooted at one controlled directory (<app data>/evidence). Every path is checked to stay
inside it (path traversal is rejected), so neither writing nor retention cleanup can ever reach a
project file or an arbitrary path taken from evidence metadata.

Retention (cleanup): deterministic and crash-safe - it only deletes files under the root, never
touches run state, never deletes artifacts of runs that are still active (resumable) or files an
active run references, and applies, in this order: maximum age, maximum bytes per finished run,
maximum total bytes (oldest first).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.evidence.model import Artifact

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_RUN_DIR = re.compile(r"^run-(\d+|x)$")
MEDIA_EXT = {"text/plain": "log", "image/png": "png", "image/jpeg": "jpg", "application/json": "json"}


def _safe(name: str, n: int = 40) -> str:
    s = _SAFE.sub("_", str(name or "")).strip("._")[:n]
    return s or "x"


def inside(root: str | Path, path: str | Path) -> bool:
    """True only if `path` resolves to a location strictly inside `root` (no traversal, no escape)."""
    try:
        r = os.path.realpath(os.path.abspath(str(root)))
        p = os.path.realpath(os.path.abspath(str(path)))
        return os.path.commonpath([r, p]) == r and p != r
    except (ValueError, OSError):
        return False


def _dir_bytes(d: Path, limit_files: int = 20000) -> int:
    total = 0
    if not d.is_dir():
        return 0
    for i, (dirpath, _dirs, files) in enumerate(os.walk(d, followlinks=False)):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
        if i > limit_files:
            break
    return total


class ArtifactStore:
    def __init__(self, root: str, run_id: int | None, phase_id: str, attempt: int, *, max_artifacts: int,
                 max_bytes: int, existing: int = 0, max_run_bytes: int = 0):
        self.root = Path(root) if root else None
        self.run_dir = (self.root / f"run-{run_id if run_id is not None else 'x'}") if self.root else None
        self.dir: Path | None = (self.run_dir / _safe(phase_id) / f"a{int(attempt)}") if self.run_dir else None
        if self.dir is not None and not inside(self.root, self.dir):  # never reachable with _safe(); defence
            self.dir = None
        self.phase_id, self.attempt = phase_id, attempt
        self.max_artifacts, self.max_bytes = max(0, max_artifacts), max(1024, max_bytes)
        self.max_run_bytes = max(0, int(max_run_bytes))  # 0 = unlimited
        self.count = existing
        self.dropped = 0
        self.drop_reasons: list[str] = []
        self._run_used: int | None = None

    def _next(self, label: str, ext: str) -> tuple[str, Path] | None:
        if self.dir is None or self.count >= self.max_artifacts:
            self.dropped += 1
            if self.dir is not None:
                self.drop_reasons.append(f"artifact limit ({self.max_artifacts}) reached")
            return None
        self.count += 1
        aid = f"art-{_safe(self.phase_id)}-a{self.attempt}-{self.count:02d}"
        path = self.dir / f"{aid}-{_safe(label)}.{_safe(ext, 8)}"
        if not inside(self.root, path):
            self.dropped += 1
            self.drop_reasons.append("artifact path outside the evidence root")
            return None
        return aid, path

    def _run_budget_ok(self, n: int) -> bool:
        if not self.max_run_bytes or self.run_dir is None:
            return True
        if self._run_used is None:
            self._run_used = _dir_bytes(self.run_dir)
        if self._run_used + n > self.max_run_bytes:
            self.dropped += 1
            self.drop_reasons.append(f"per-run artifact budget ({self.max_run_bytes} bytes) reached")
            return False
        self._run_used += n
        return True

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def save_text(self, kind: str, label: str, text: str, description: str, capability_id: str) -> Artifact | None:
        """Store text (tail-bounded). Returns None when artifacts are disabled or a budget is used up."""
        data = text.encode("utf-8", errors="replace")
        truncated = len(data) > self.max_bytes
        if truncated:
            data = b"...[truncated: showing the last %d bytes]...\n" % self.max_bytes + data[-self.max_bytes:]
        if self.dir is None or self.count >= self.max_artifacts:
            self._next(label, "log")  # counts the drop
            return None
        if not self._run_budget_ok(len(data)):
            return None
        slot = self._next(label, "log")
        if slot is None:
            return None
        aid, path = slot
        self._write(path, data)
        return Artifact(aid, kind, str(path), description[:300], capability_id, len(data), truncated, time.time())

    def save_bytes(self, kind: str, label: str, data: bytes, media_type: str, description: str, capability_id: str,
                   metadata: dict[str, Any] | None = None, ext: str = "") -> Artifact | None:
        """Store a binary artifact (e.g. an image). Binary data is never truncated: too large -> not stored."""
        if len(data) > self.max_bytes:
            self.dropped += 1
            self.drop_reasons.append(f"{label}: {len(data)} bytes exceed the artifact size limit ({self.max_bytes})")
            return None
        if self.dir is None or self.count >= self.max_artifacts:
            self._next(label, "bin")
            return None
        if not self._run_budget_ok(len(data)):
            return None
        slot = self._next(label, ext or MEDIA_EXT.get(media_type, "bin"))
        if slot is None:
            return None
        aid, path = slot
        self._write(path, data)
        return Artifact(aid, kind, str(path), description[:300], capability_id, len(data), False, time.time(),
                        media_type=media_type, metadata=dict(metadata or {}))


# ------------------------------------------------------------ retention

@dataclass(frozen=True)
class RetentionPolicy:
    max_age_days: float = 14.0  # 0 = no age limit
    max_run_bytes: int = 200_000_000  # per finished run; 0 = unlimited
    max_total_bytes: int = 1_000_000_000  # everything under the root; 0 = unlimited

    @classmethod
    def from_settings(cls, o: Any) -> "RetentionPolicy":
        return cls(max(0.0, float(getattr(o, "evidence_artifact_max_age_days", 14))),
                   max(0, int(getattr(o, "evidence_artifact_max_run_bytes", 200_000_000))),
                   max(0, int(getattr(o, "evidence_artifact_max_total_bytes", 1_000_000_000))))


@dataclass
class CleanupReport:
    scanned: int = 0
    deleted: list[str] = field(default_factory=list)
    freed: int = 0
    protected: int = 0  # files kept because an active run owns or references them
    errors: list[str] = field(default_factory=list)
    total_after: int = 0

    def summary(self) -> str:
        return (f"evidence artifact cleanup: {len(self.deleted)} file(s) deleted ({self.freed} bytes), "
                f"{self.protected} protected, {self.total_after} bytes kept"
                + (f", {len(self.errors)} error(s)" if self.errors else ""))


def _files(root: Path) -> list[tuple[Path, int, float, str]]:
    """(path, size, mtime, owner) of the files under the root; owner = run id string, or "" for other dirs."""
    out: list[tuple[Path, int, float, str]] = []
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(dirpath, d))]
        rel = Path(dirpath).relative_to(root).parts
        owner = ""
        if rel and _RUN_DIR.match(rel[0]):
            owner = rel[0][4:]
        for f in files:
            p = Path(dirpath) / f
            if os.path.islink(p) or not inside(root, p):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((p, st.st_size, st.st_mtime, owner))
    return sorted(out, key=lambda x: (x[2], str(x[0])))  # oldest first, deterministic


def cleanup(root: str | Path, *, protected_runs: set[int] | set[str], referenced: set[str] | None = None,
            policy: RetentionPolicy = RetentionPolicy(), now: float | None = None) -> CleanupReport:
    """Apply the retention policy under `root` only. Artifacts of protected (active) runs and referenced
    paths are never deleted. Deleting a file never changes run state, so a crash here is harmless."""
    rep = CleanupReport()
    base = Path(root)
    if not str(root) or not base.is_dir():
        return rep
    base = Path(os.path.realpath(base))
    now = time.time() if now is None else now
    keep_runs = {str(x) for x in protected_runs}
    refs = {os.path.normcase(os.path.realpath(x)) for x in (referenced or set()) if x and inside(base, x)}
    files = _files(base)
    rep.scanned = len(files)

    def protected(p: Path, owner: str) -> bool:
        return owner in keep_runs or os.path.normcase(os.path.realpath(p)) in refs

    alive: list[tuple[Path, int, float, str]] = []
    for item in files:
        p, size, mtime, owner = item
        if protected(p, owner):
            rep.protected += 1
            alive.append(item)
            continue
        if policy.max_age_days and now - mtime > policy.max_age_days * 86400:
            _delete(p, size, rep)
        else:
            alive.append(item)
    if policy.max_run_bytes:
        by_run: dict[str, list[tuple[Path, int, float, str]]] = {}
        for item in alive:
            if item[3] and not protected(item[0], item[3]):
                by_run.setdefault(item[3], []).append(item)
        for run, items in sorted(by_run.items()):
            total = sum(i[1] for i in items)
            for p, size, _m, _o in items:  # oldest first
                if total <= policy.max_run_bytes:
                    break
                if _delete(p, size, rep):
                    total -= size
        alive = [i for i in alive if str(i[0]) not in set(rep.deleted)]
    total = sum(i[1] for i in alive)
    if policy.max_total_bytes and total > policy.max_total_bytes:
        for p, size, _m, owner in alive:
            if total <= policy.max_total_bytes:
                break
            if not protected(p, owner) and _delete(p, size, rep):
                total -= size
    rep.total_after = sum(i[1] for i in alive if str(i[0]) not in set(rep.deleted))
    _remove_empty_dirs(base)
    return rep


def _delete(p: Path, size: int, rep: CleanupReport) -> bool:
    try:
        p.unlink()
    except FileNotFoundError:
        return True
    except OSError as exc:
        rep.errors.append(f"{p.name}: {exc}"[:200])
        return False
    rep.deleted.append(str(p))
    rep.freed += size
    return True


def _remove_empty_dirs(base: Path) -> None:
    for dirpath, dirs, files in sorted(os.walk(base, topdown=False, followlinks=False), key=lambda x: -len(x[0])):
        d = Path(dirpath)
        if d == base or not inside(base, d) or os.path.islink(d):
            continue
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
