"""Process identity for crash-safe cleanup of evidence resources (no third-party dependency).

A pid alone is not an identity: after a restart it may belong to an unrelated process. A stale
resource is only killed when the pid is alive AND its creation time matches the one recorded when
AI Studio started it; otherwise it is left alone and reported.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

IS_WINDOWS = sys.platform == "win32"
_STILL_ACTIVE = 259


def _win_times(pid: int) -> tuple[bool, float | None]:
    """(alive, creation time as epoch seconds) via OpenProcess/GetProcessTimes."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False, None
    try:
        code = wintypes.DWORD()
        alive = bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == _STILL_ACTIVE
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        if not k32.GetProcessTimes(handle, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
            return alive, None
        ticks = (c.dwHighDateTime << 32) | c.dwLowDateTime  # 100 ns since 1601-01-01
        return alive, ticks / 1e7 - 11644473600.0
    finally:
        k32.CloseHandle(handle)


def _posix_times(pid: int) -> tuple[bool, float | None]:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False, None
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/stat", encoding="ascii") as f:
            btime = next(int(line.split()[1]) for line in f if line.startswith("btime"))
        return True, btime + start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return True, None  # alive, identity unverifiable


def process_info(pid: int | None) -> tuple[bool, float | None]:
    if not pid or pid <= 0:
        return False, None
    try:
        return _win_times(pid) if IS_WINDOWS else _posix_times(pid)
    except Exception:
        return False, None


def start_time(pid: int | None) -> float | None:
    return process_info(pid)[1]


def is_alive(pid: int | None) -> bool:
    return process_info(pid)[0]


def kill_tree(pid: int) -> None:
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        import signal
        try:
            os.killpg(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass


def reap(pid: int | None, recorded_start: float | None, tolerance: float = 2.0) -> str:
    """Stop a stale evidence process left by a crashed run. Returns gone | killed | reused | unverified."""
    alive, started = process_info(pid)
    if not alive:
        return "gone"
    if started is None or recorded_start is None:
        return "unverified"  # cannot prove it is ours: never kill a process we cannot identify
    if abs(started - recorded_start) > tolerance:
        return "reused"  # the pid now belongs to another process
    kill_tree(int(pid))
    for _ in range(50):
        if not is_alive(pid):
            break
        time.sleep(0.1)
    return "killed"
