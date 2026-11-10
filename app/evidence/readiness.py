"""Generic readiness probes for evidence resources (declared services / runtimes).

A readiness declaration is capability METADATA, validated when it is discovered and never turned into
a shell command:

    {"kind": "process_alive",   "settle_seconds": 1}
    {"kind": "stdout_contains", "text": "READY"}
    {"kind": "tcp_port",        "host": "127.0.0.1", "port": 5173}
    {"kind": "http_status",     "url": "http://127.0.0.1:5173/health", "expected_status": 200}
    {"kind": "file_exists",     "path": "logs/ready.flag"}          (repository-relative)

Every probe also fails fast when the process exits before it is ready. Network probes are limited to
this machine (localhost / 127.0.0.1 / ::1), follow no redirects, use no proxy, send no credentials and
read at most a small bounded response. Timeouts are bounded by the caller's readiness budget.
"""
from __future__ import annotations

import asyncio
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

READINESS_KINDS = ("process_alive", "stdout_contains", "tcp_port", "http_status", "file_exists")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
MAX_TEXT = 200
MAX_BODY = 64 * 1024
POLL = 0.1


def _loopback(host: str) -> bool:
    return str(host or "").strip().lower().strip("[]") in LOOPBACK_HOSTS


def _port(v: Any) -> int | None:
    try:
        p = int(v)
    except (TypeError, ValueError):
        return None
    return p if 1 <= p <= 65535 else None


def relative_path_ok(rel: str) -> bool:
    p = str(rel or "").replace("\\", "/")
    if not p or p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        return False
    parts = [x for x in p.split("/") if x not in ("", ".")]
    return bool(parts) and ".." not in parts and parts[0].lower() != ".git"


def check_local_url(url: str) -> str:
    """"" if the URL is a safe local probe target, otherwise why not."""
    try:
        u = urllib.parse.urlsplit(str(url or "").strip())
    except ValueError:
        return "invalid URL"
    if u.scheme not in ("http", "https"):
        return "only http:// or https:// URLs can be probed"
    if u.username or u.password:
        return "URLs with credentials are not allowed"
    if not _loopback(u.hostname or ""):
        return f"network probes are limited to this machine (localhost), not {u.hostname!r}"
    try:
        if u.port is not None and _port(u.port) is None:
            return "invalid port"
    except ValueError:
        return "invalid port"
    if len(url) > 500:
        return "URL too long"
    return ""


def redact_url(url: str) -> str:
    """For logs: scheme://host:port/path (query strings and fragments are never logged)."""
    try:
        u = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((u.scheme, u.netloc.split("@")[-1], u.path, "", ""))
    except ValueError:
        return "(invalid url)"


def parse_readiness(raw: Any, *, legacy_marker: str = "", default_timeout: float = 60) -> tuple[dict[str, Any], str]:
    """Normalized readiness spec and "" - or ({}, why it is invalid). Without a declaration: the legacy
    `ready_output` marker (stdout_contains), else process_alive."""
    if raw in (None, "", {}):
        if legacy_marker:
            return {"kind": "stdout_contains", "text": str(legacy_marker)[:MAX_TEXT],
                    "timeout_seconds": default_timeout}, ""
        return {"kind": "process_alive", "settle_seconds": 1.0, "timeout_seconds": default_timeout}, ""
    if not isinstance(raw, dict):
        return {}, "readiness must be an object"
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in READINESS_KINDS:
        return {}, f"unknown readiness kind {kind!r} (expected one of {', '.join(READINESS_KINDS)})"
    try:
        timeout = float(raw.get("timeout_seconds") or default_timeout)
    except (TypeError, ValueError):
        return {}, "timeout_seconds must be a number"
    spec: dict[str, Any] = {"kind": kind, "timeout_seconds": max(0.5, min(timeout, 600.0))}
    if kind == "process_alive":
        try:
            spec["settle_seconds"] = max(0.1, min(float(raw.get("settle_seconds") or 1.0), 30.0))
        except (TypeError, ValueError):
            return {}, "settle_seconds must be a number"
    elif kind == "stdout_contains":
        text = str(raw.get("text") or legacy_marker or "")
        if not 0 < len(text) <= MAX_TEXT:
            return {}, f"stdout_contains needs a text of 1-{MAX_TEXT} characters"
        spec["text"] = text
    elif kind == "tcp_port":
        host = str(raw.get("host") or "127.0.0.1").strip()
        if not _loopback(host):
            return {}, f"network probes are limited to this machine (localhost), not {host!r}"
        port = _port(raw.get("port"))
        if port is None:
            return {}, "tcp_port needs a port between 1 and 65535"
        spec.update(host=host.strip("[]"), port=port)
    elif kind == "http_status":
        url = str(raw.get("url") or "").strip()
        why = check_local_url(url)
        if why:
            return {}, why
        try:
            expected = int(raw.get("expected_status") or 200)
        except (TypeError, ValueError):
            return {}, "expected_status must be an HTTP status code"
        if not 100 <= expected <= 599:
            return {}, "expected_status must be an HTTP status code (100-599)"
        spec.update(url=url, expected_status=expected)
    elif kind == "file_exists":
        path = str(raw.get("path") or "")
        if len(path) > 300 or not relative_path_ok(path):
            return {}, "file_exists needs a repository-relative path outside .git"
        spec["path"] = path
    return spec, ""


def describe(spec: dict[str, Any]) -> str:
    k = spec.get("kind")
    if k == "stdout_contains":
        return f"its output shows {spec.get('text')!r}"
    if k == "tcp_port":
        return f"it accepts TCP connections on {spec.get('host')}:{spec.get('port')}"
    if k == "http_status":
        return f"{redact_url(spec.get('url', ''))} answers HTTP {spec.get('expected_status')}"
    if k == "file_exists":
        return f"the file {spec.get('path')} exists"
    return f"it keeps running for {spec.get('settle_seconds', 1)}s"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401 - never follow redirects
        return None


def http_get(url: str, timeout: float, max_body: int = MAX_BODY) -> tuple[int | None, str, str]:
    """(status, bounded body text, error) of ONE local GET: no proxy, no redirects, no credentials."""
    why = check_local_url(url)
    if why:
        return None, "", why
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "AIStudio-evidence/1"})
    try:
        with opener.open(req, timeout=max(0.2, timeout)) as resp:  # noqa: S310 - validated local URL
            return int(resp.status), resp.read(max_body).decode("utf-8", errors="replace"), ""
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(max_body).decode("utf-8", errors="replace")
        except Exception:
            pass
        return int(exc.code), body, ""
    except Exception as exc:
        return None, "", str(getattr(exc, "reason", exc))[:200]


async def tcp_open(host: str, port: int, timeout: float) -> bool:
    try:
        _r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=max(0.1, timeout))
    except (OSError, asyncio.TimeoutError):
        return False
    w.close()
    try:
        await asyncio.wait_for(w.wait_closed(), timeout=1)
    except Exception:
        pass
    return True


async def wait_ready(handle: Any, spec: dict[str, Any], timeout: float, repo_path: str = "") -> tuple[str, str]:
    """ready | exited | timeout for a started process handle (running, exit_code, seen(text))."""
    kind = spec.get("kind", "process_alive")
    timeout = max(0.2, min(float(timeout), float(spec.get("timeout_seconds") or timeout)))
    start = time.monotonic()
    last_error = ""
    while True:
        elapsed = time.monotonic() - start
        if kind == "stdout_contains" and handle.seen(spec.get("text", "")):
            return "ready", f"ready marker {spec.get('text')!r} seen"
        if not handle.running:
            return "exited", f"exited with code {handle.exit_code} before it was ready"
        if kind == "process_alive" and elapsed >= min(float(spec.get("settle_seconds", 1.0)), timeout):
            return "ready", f"still running after {min(float(spec.get('settle_seconds', 1.0)), timeout):.1f}s"
        if kind == "tcp_port" and await tcp_open(spec["host"], int(spec["port"]), min(1.0, timeout)):
            return "ready", f"{spec['host']}:{spec['port']} accepts connections"
        if kind == "http_status":
            status, _body, err = await asyncio.to_thread(http_get, spec["url"], min(2.0, max(0.2, timeout - elapsed)))
            if status == int(spec.get("expected_status", 200)):
                return "ready", f"{redact_url(spec['url'])} answered HTTP {status}"
            last_error = f"HTTP {status}" if status is not None else err
        if kind == "file_exists" and repo_path:
            full = os.path.abspath(os.path.join(repo_path, *spec["path"].replace("\\", "/").split("/")))
            if full.startswith(os.path.abspath(repo_path) + os.sep) and os.path.exists(full):
                return "ready", f"{spec['path']} exists"
        if time.monotonic() - start >= timeout:
            detail = f"not ready within {timeout:.0f}s ({describe(spec)} never happened)"
            return "timeout", detail + (f"; last probe: {last_error}" if last_error else "")
        await asyncio.sleep(POLL)
