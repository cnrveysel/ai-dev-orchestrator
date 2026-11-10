"""Windows window capture for visual evidence (a CaptureBackend of the visual provider).

It captures ONLY a window owned by the process tree of a resource AI Studio itself started for evidence
(the running resource's pid and its descendants) - never the desktop, never another application. The
window is asked to render itself (PrintWindow), so it works even when the window is covered or placed
off-screen. The result is an honest record of what the window showed: a PNG plus its size. It is not a
judgment of the content (that needs image analysis or a person).

No third-party dependency: ctypes (user32/gdi32/kernel32) and a small PNG encoder (zlib).
"""
from __future__ import annotations

import asyncio
import struct
import sys
import zlib
from typing import Any

from app.evidence.providers.visual import CapturedImage, CaptureBackend

MAX_SIDE = 8192
PW_RENDERFULLCONTENT = 2


def encode_png(bgra: bytes, width: int, height: int) -> bytes:
    """32-bit BGRA rows (top-down) -> an RGB PNG."""
    rows = []
    stride = width * 4
    for y in range(height):
        row = bgra[y * stride:(y + 1) * stride]
        rgb = bytearray(width * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = row[2::4], row[1::4], row[0::4]
        rows.append(b"\x00" + bytes(rgb))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def _api() -> Any:
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    u32 = ctypes.WinDLL("user32", use_last_error=True)
    g32 = ctypes.WinDLL("gdi32", use_last_error=True)
    return ctypes, wintypes, k32, u32, g32


def process_tree(root_pid: int) -> set[int]:
    """root_pid and all its descendants (Toolhelp32 snapshot)."""
    ctypes, wintypes, k32, _u32, _g32 = _api()

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return {root_pid}
    parents: dict[int, list[int]] = {}
    try:
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            parents.setdefault(int(e.th32ParentProcessID), []).append(int(e.th32ProcessID))
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    tree, todo = {root_pid}, [root_pid]
    while todo:
        for child in parents.get(todo.pop(), []):
            if child not in tree and child != 0:
                tree.add(child)
                todo.append(child)
    return tree


def windows_of(pids: set[int]) -> list[tuple[int, int, int, str]]:
    """Visible top-level windows of these processes: (hwnd, width, height, title), largest first."""
    ctypes, wintypes, _k32, u32, _g32 = _api()
    found: list[tuple[int, int, int, str]] = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _lp: int) -> bool:
        pid = wintypes.DWORD()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and u32.IsWindowVisible(hwnd):
            rect = wintypes.RECT()
            if u32.GetWindowRect(hwnd, ctypes.byref(rect)):
                w, h = rect.right - rect.left, rect.bottom - rect.top
                if 1 < w <= MAX_SIDE and 1 < h <= MAX_SIDE:
                    n = u32.GetWindowTextLengthW(hwnd)
                    buf = ctypes.create_unicode_buffer(min(n, 200) + 1)
                    u32.GetWindowTextW(hwnd, buf, len(buf))
                    found.append((int(hwnd), w, h, buf.value))
        return True
    u32.EnumWindows(proto(visit), 0)
    return sorted(found, key=lambda x: -(x[1] * x[2]))


def capture_hwnd(hwnd: int, width: int, height: int) -> bytes | None:
    """BGRA pixels of one window rendered by itself (PrintWindow), or None."""
    ctypes, wintypes, _k32, u32, g32 = _api()

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                    ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]
    for fn, res in (("GetDC", wintypes.HDC), ("CreateCompatibleDC", wintypes.HDC),
                    ("CreateCompatibleBitmap", wintypes.HBITMAP), ("SelectObject", wintypes.HGDIOBJ)):
        getattr(u32 if fn == "GetDC" else g32, fn).restype = res
    u32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    g32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    g32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    g32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    g32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                              ctypes.c_void_p, wintypes.UINT]
    g32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    g32.DeleteDC.argtypes = [wintypes.HDC]
    u32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    screen = u32.GetDC(None)
    mem = g32.CreateCompatibleDC(screen)
    bmp = g32.CreateCompatibleBitmap(screen, width, height)
    old = g32.SelectObject(mem, bmp)
    try:
        if not u32.PrintWindow(hwnd, mem, PW_RENDERFULLCONTENT):
            return None
        bmi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = ctypes.create_string_buffer(width * height * 4)
        lines = g32.GetDIBits(mem, bmp, 0, height, buf, ctypes.byref(bmi), 0)
        return bytes(buf.raw) if lines == height else None
    finally:
        g32.SelectObject(mem, old)
        g32.DeleteObject(bmp)
        g32.DeleteDC(mem)
        u32.ReleaseDC(None, screen)


class WindowsWindowCapture(CaptureBackend):
    name = "windows-window"

    def available(self) -> tuple[bool, str]:
        if sys.platform != "win32":
            return False, "window capture is implemented for Windows only"
        try:
            _api()
        except Exception as exc:  # pragma: no cover - no user32/gdi32
            return False, f"window capture unavailable: {exc}"
        return True, "captures a window of a process AI Studio started for evidence (never the desktop)"

    async def capture(self, handle: Any, *, timeout: float) -> CapturedImage | None:
        pid = getattr(handle, "pid", None)
        if not pid or not getattr(handle, "running", False):
            return None
        return await asyncio.wait_for(asyncio.to_thread(self._capture, int(pid)), timeout=max(1.0, timeout))

    @staticmethod
    def _capture(pid: int) -> CapturedImage | None:
        wins = windows_of(process_tree(pid))
        for hwnd, w, h, title in wins:
            pixels = capture_hwnd(hwnd, w, h)
            if pixels:
                return CapturedImage(encode_png(pixels, w, h), w, h,
                                     description=f"window {title!r}" if title else "untitled window",
                                     metadata={"window_title": title[:200], "pid": pid})
        return None
