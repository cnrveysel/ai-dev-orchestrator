"""Fresh-log reading: only the part of a log file that provably belongs to the CURRENT run.

A tool that writes its results to a log file (instead of, or besides, its output) leaves that file behind:
the next run may fail before writing anything, and an old "success" line would still be there. So a
provider that falls back to a log snapshots it BEFORE it launches the tool and afterwards reads only what
the new run wrote:

  missing before, present after                     -> the whole file (created by this run)
  same file grown (its old head is unchanged)       -> only the bytes appended after the old size
  replaced / rotated (head changed or file shrank)  -> the whole file, if it was written after the launch
  unchanged, or last written before the launch      -> nothing (stale)

Nothing is ever deleted or rotated in the project: the check is read-only. Reading is bounded.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

HEAD_BYTES = 4096
MAX_READ_BYTES = 8 * 1024 * 1024


def _head(path: str, n: int = HEAD_BYTES) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read(n)).hexdigest()
    except OSError:
        return ""


@dataclass
class FreshLog:
    path: str
    existed: bool = False
    size: int = 0
    mtime: float = 0.0
    head: str = ""  # hash of the first HEAD_BYTES (or fewer) bytes before the launch
    head_len: int = 0
    taken_at: float = 0.0

    @classmethod
    def snapshot(cls, path: str) -> "FreshLog":
        """Record the file's state; call this BEFORE launching the tool that writes it."""
        s = cls(path, taken_at=time.time())
        try:
            st = os.stat(path)
        except OSError:
            return s
        s.existed, s.size, s.mtime = True, st.st_size, st.st_mtime
        s.head_len = min(st.st_size, HEAD_BYTES)
        s.head = _head(path, s.head_len)
        return s

    def fresh_text(self, limit: int = MAX_READ_BYTES) -> tuple[str, str]:
        """(text written by the current run, how it was established). ("", why) when nothing is provably new."""
        try:
            st = os.stat(self.path)
        except OSError:
            return "", "no log file"
        if st.st_mtime < self.taken_at - 0.01 and (not self.existed or st.st_mtime <= self.mtime):
            return "", "the log was last written before this run started (stale)"
        if not self.existed:
            return self._read(0, limit), "created by this run"
        if st.st_size == self.size and st.st_mtime == self.mtime:
            return "", "the log is unchanged since before this run (stale)"
        grown = st.st_size >= self.size and _head(self.path, self.head_len) == self.head
        if grown:
            if st.st_size == self.size:
                return "", "nothing was appended by this run"
            return self._read(self.size, limit), "appended by this run"
        if st.st_mtime >= self.taken_at - 0.01:
            return self._read(0, limit), "replaced by this run"
        return "", "the log changed but not during this run"

    def _read(self, offset: int, limit: int) -> str:
        try:
            with open(self.path, "rb") as f:
                f.seek(offset)
                return f.read(limit).decode("utf-8", errors="replace")
        except OSError:
            return ""
