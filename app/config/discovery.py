"""Discovery and validation of the two executables the app depends on.

The worker must be the dedicated Codex NInfer executable. The normal `codex`
command on this machine is an interactive launcher/menu (or the npm OpenAI
Codex shim) and must never be used as the worker.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


class ExecutableRejected(ValueError):
    pass


@dataclass
class Candidate:
    path: str
    source: str
    mtime: float
    size: int


def _env_path(var: str, *parts: str) -> Path | None:
    base = os.environ.get(var)
    return Path(base, *parts) if base else None


def _norm(p: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


# ------------------------------------------------------------------ Claude

def discover_claude() -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    paths: list[tuple[Path | None, str]] = [
        (Path(shutil.which("claude")) if shutil.which("claude") else None, "PATH"),
        (_env_path("USERPROFILE", ".local", "bin", "claude.exe"), "~/.local/bin"),
        (_env_path("APPDATA", "npm", "node_modules", "@anthropic-ai", "claude-code", "bin", "claude.exe"), "npm"),
    ]
    for p, src in paths:
        if p is None or not p.is_file():
            continue
        if p.suffix.lower() in (".cmd", ".bat"):
            continue  # batch shims are not launched directly
        key = _norm(p)
        if key in seen:
            continue
        seen.add(key)
        st = p.stat()
        out.append(Candidate(str(p), src, st.st_mtime, st.st_size))
    return out


def validate_claude_executable(path: str) -> str:
    if not path:
        raise ExecutableRejected("Claude Code CLI executable is not configured")
    p = Path(path)
    if not p.is_file():
        raise ExecutableRejected(f"Claude Code CLI not found: {path}")
    if p.suffix.lower() in (".cmd", ".bat"):
        raise ExecutableRejected("select the claude .exe, not a batch shim")
    if "claude" not in p.name.lower():
        raise ExecutableRejected(f"{p.name} does not look like the Claude Code CLI")
    return str(p)


# ------------------------------------------------------------------ Codex NInfer

def normal_codex_paths() -> set[str]:
    """Known locations of normal Codex / the interactive launcher (forbidden)."""
    bad: set[str] = set()
    which = shutil.which("codex")
    if which:
        bad.add(_norm(which))
    for var, parts in [
        ("APPDATA", ("npm", "codex")),
        ("APPDATA", ("npm", "codex.cmd")),
        ("APPDATA", ("npm", "codex.ps1")),
        ("USERPROFILE", ("bin", "codex-menu.cmd")),
    ]:
        p = _env_path(var, *parts)
        if p:
            bad.add(_norm(p))
    return bad


def validate_worker_executable(path: str) -> str:
    """Raise ExecutableRejected unless `path` is a dedicated codex-ninfer exe.
    There is deliberately no fallback to any other executable."""
    if not path:
        raise ExecutableRejected("Codex NInfer executable is not configured. "
                                 "Select your codex-ninfer.exe in Settings.")
    p = Path(path)
    if _norm(p) in normal_codex_paths():
        raise ExecutableRejected(f"{path} is the normal `codex` launcher, not Codex NInfer")
    if not p.is_file():
        raise ExecutableRejected(f"Codex NInfer executable not found: {path}")
    name = p.name.lower()
    if p.suffix.lower() != ".exe":
        raise ExecutableRejected(f"worker must be a .exe (got {p.name}); launchers/scripts are not allowed")
    if "ninfer" not in name:
        raise ExecutableRejected(
            f"{p.name} is not a codex-ninfer executable. The normal codex / launcher is never used as worker.")
    if "npm" in [part.lower() for part in p.parts] or "node_modules" in _norm(p):
        raise ExecutableRejected(f"{path} is inside an npm installation (normal Codex)")
    return str(p)


def validate_codex_executable(path: str) -> str:
    """V5 worker profiles: a dedicated Codex build (e.g. codex-ninfer.exe, codex-lmstudio.exe). The same
    protections as validate_worker_executable apply - never the normal `codex` launcher / menu / npm shim, never
    a script - except that the file name does not have to contain "ninfer"."""
    if not path:
        raise ExecutableRejected("the worker profile has no Codex executable configured")
    p = Path(path)
    if _norm(p) in normal_codex_paths():
        raise ExecutableRejected(f"{path} is the normal `codex` launcher; a worker profile needs a dedicated build")
    if not p.is_file():
        raise ExecutableRejected(f"Codex executable not found: {path}")
    if p.suffix.lower() != ".exe":
        raise ExecutableRejected(f"worker must be a .exe (got {p.name}); launchers/scripts are not allowed")
    if p.stem.lower() in ("codex", "codex-menu"):
        raise ExecutableRejected(f"{p.name} is the normal `codex` launcher name; use a dedicated Codex build")
    if "npm" in [part.lower() for part in p.parts] or "node_modules" in _norm(p):
        raise ExecutableRejected(f"{path} is inside an npm installation (normal Codex)")
    return str(p)


def validate_executable_for_policy(path: str, policy: str = "ninfer") -> str:
    """"ninfer" = the V4 rule (legacy worker settings), "codex" = any dedicated Codex build (worker profiles)."""
    return validate_codex_executable(path) if policy == "codex" else validate_worker_executable(path)


def discover_codex_ninfer() -> list[Candidate]:
    """Candidates, newest first. Never includes the normal codex command."""
    home = os.environ.get("USERPROFILE", str(Path.home()))
    dirs = [
        (_env_path("LOCALAPPDATA", "CodexNInfer", "bin"), "%LOCALAPPDATA%\\CodexNInfer\\bin"),
        (Path(home, "bin", "ninfer"), "~/bin/ninfer"),
        (Path(home, "bin"), "~/bin"),
        (_env_path("LOCALAPPDATA", "Programs", "CodexNInfer"), "%LOCALAPPDATA%\\Programs\\CodexNInfer"),
    ]
    which = shutil.which("codex-ninfer")
    seen: set[str] = set()
    out: list[Candidate] = []

    def consider(p: Path, src: str) -> None:
        key = _norm(p)
        if key in seen or not p.is_file():
            return
        try:
            validate_worker_executable(str(p))
        except ExecutableRejected:
            return
        # Backup builds such as codex-ninfer-before-xyz.exe are not offered.
        if p.stem.lower() != "codex-ninfer":
            return
        seen.add(key)
        st = p.stat()
        out.append(Candidate(str(p), src, st.st_mtime, st.st_size))

    for d, src in dirs:
        if d is not None and d.is_dir():
            consider(d / "codex-ninfer.exe", src)
    if which:
        consider(Path(which), "PATH")
    out.sort(key=lambda c: c.mtime, reverse=True)
    return out


def discover_codex_builds() -> list[Candidate]:
    """V5 worker profiles: dedicated Codex builds such as codex-ninfer.exe / codex-lmstudio.exe, newest first.
    Backup builds (codex-ninfer-before-x.exe, codex-lmstudio-v1.exe, ...) and the normal launcher are not offered."""
    import re
    home = os.environ.get("USERPROFILE", str(Path.home()))
    dirs = [Path(home, "bin", "ninfer"), Path(home, "bin"), _env_path("LOCALAPPDATA", "CodexNInfer", "bin")]
    seen: set[str] = set()
    out: list[Candidate] = []
    for d in dirs:
        if d is None or not d.is_dir():
            continue
        for p in sorted(d.glob("codex-*.exe")):
            if not re.fullmatch(r"codex-[a-z]+", p.stem.lower()) or _norm(p) in seen:
                continue
            try:
                validate_codex_executable(str(p))
            except ExecutableRejected:
                continue
            seen.add(_norm(p))
            st = p.stat()
            out.append(Candidate(str(p), str(d), st.st_mtime, st.st_size))
    out.sort(key=lambda c: c.mtime, reverse=True)
    return out


# ------------------------------------------------------------------ Cline (V5.1 worker runtime)

CLINE_NATIVE_PARTS = ("node_modules", "cline", "node_modules", "@cline", "cli-windows-x64", "bin", "cline.exe")


def validate_cline_executable(path: str) -> str:
    """The native Cline CLI binary (npm installs it at npm\\node_modules\\cline\\node_modules\\@cline\\
    cli-windows-x64\\bin\\cline.exe). The npm `cline` / `cline.cmd` shims are refused: AI Studio never launches
    batch files or scripts, and there is no fallback."""
    if not path:
        raise ExecutableRejected("the Cline profile has no Cline executable configured")
    p = Path(path)
    if not p.is_file():
        raise ExecutableRejected(f"Cline executable not found: {path}")
    if p.suffix.lower() != ".exe":
        raise ExecutableRejected(f"select the native cline.exe, not {p.name} (npm shims/scripts are not launched); "
                                 "it is under npm\\node_modules\\cline\\node_modules\\@cline\\cli-windows-x64\\bin")
    if "cline" not in p.stem.lower():
        raise ExecutableRejected(f"{p.name} does not look like the Cline CLI")
    return str(p)


def discover_cline_executables() -> list[Candidate]:
    """The native cline.exe of the npm installation (and of a `cline` found on PATH), newest first."""
    roots: list[tuple[Path, str]] = []
    npm = _env_path("APPDATA", "npm")
    if npm is not None:
        roots.append((npm, "%APPDATA%\\npm"))
    which = shutil.which("cline")
    if which:
        roots.append((Path(which).parent, "PATH"))
    seen: set[str] = set()
    out: list[Candidate] = []
    for root, src in roots:
        p = Path(root, *CLINE_NATIVE_PARTS)
        key = _norm(p)
        if key in seen or not p.is_file():
            continue
        seen.add(key)
        st = p.stat()
        out.append(Candidate(str(p), src, st.st_mtime, st.st_size))
    out.sort(key=lambda c: c.mtime, reverse=True)
    return out


def discover_codebase_memory() -> list[Candidate]:
    out = []
    for p in [_env_path("LOCALAPPDATA", "Programs", "codebase-memory-mcp", "codebase-memory-mcp.exe"),
              Path(shutil.which("codebase-memory-mcp")) if shutil.which("codebase-memory-mcp") else None]:
        if p is not None and p.is_file() and all(_norm(c.path) != _norm(p) for c in out):
            st = p.stat()
            out.append(Candidate(str(p), "auto", st.st_mtime, st.st_size))
    return out


def autofill(settings) -> list[str]:
    """Fill empty executable settings from discovery. Returns log messages."""
    msgs = []
    if not settings.claude.executable:
        c = discover_claude()
        if c:
            settings.claude.executable = c[0].path
            msgs.append(f"Discovered Claude Code CLI: {c[0].path}")
    if not settings.worker.executable:
        c = discover_codex_ninfer()
        if c:
            settings.worker.executable = c[0].path
            msgs.append(f"Discovered Codex NInfer worker: {c[0].path}")
        else:
            msgs.append("No codex-ninfer executable found; select it in Settings.")
    if not settings.integrations.codebase_memory_executable:
        c = discover_codebase_memory()
        if c:
            settings.integrations.codebase_memory_executable = c[0].path
    return msgs
