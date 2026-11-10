"""codebase-memory-mcp status probe (the worker uses it via its own MCP config;
this app only reports availability / index coverage and, for V2, makes sure the
repository root is indexed and knows its exact project name)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from app.workers.process_runner import ProcessRunner

ALREADY_INDEXED = "already indexed"
# index_repository result statuses meaning the graph for the repository is usable.
INDEX_OK_STATUSES = frozenset({"indexed", "already_indexed", "up_to_date", "unchanged"})
_LIST_PAGE = 500  # list_projects maximum page size (0.11.0); the tool's default is only 50
_LIST_MAX_PAGES = 20
_ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/])")


@dataclass
class CodebaseMemoryStatus:
    available: bool
    version: str = ""
    repo_indexed: bool | None = None
    detail: str = ""
    project: str = ""  # exact codebase-memory project name whose root_path is this repository

    def describe(self) -> str:
        if not self.available:
            return f"unavailable ({self.detail})" if self.detail else "unavailable"
        idx = {True: "repo indexed", False: "repo NOT indexed", None: "index unknown"}[self.repo_indexed]
        if self.project:
            idx += f" as {self.project}"
        return f"{self.version or 'available'} - {idx}"


@dataclass
class IndexResult:
    """Interpretation of one `cli index_repository` call."""
    ok: bool
    status: str = ""
    project: str = ""  # project name reported by codebase-memory (if the result carries one)
    root_path: str = ""
    nodes: int | None = None
    edges: int | None = None
    detail: str = ""
    unrecognised: bool = False  # exit 0, but no result object with a status was found

    def stats(self) -> str:
        return ", ".join(f"{v} {k}" for k, v in (("nodes", self.nodes), ("edges", self.edges)) if v is not None)


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/").rstrip("/")


def _snippet(text: str, n: int = 300) -> str:
    return " ".join((text or "").split())[-n:]


def json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object embedded in `text`, in order. Text around the objects
    (log lines, protocol noise) is skipped. For an MCP result envelope
    ({"content": [{"type": "text", "text": ...}], "isError": ...}) the objects inside its
    text items follow the envelope itself."""
    out: list[dict[str, Any]] = []
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            out.append(obj)
            for t in _envelope_texts(obj):
                out += json_objects(t)
        i = text.find("{", end)
    return out


def _envelope_texts(obj: dict[str, Any]) -> list[str]:
    content = obj.get("content")
    if not isinstance(content, list):
        return []
    return [c["text"] for c in content if isinstance(c, dict) and isinstance(c.get("text"), str)]


def _texts(stdout: str) -> list[str]:
    """The tool's text output: the envelope's text items when printed with --json, else stdout."""
    texts = [t for o in json_objects(stdout) for t in _envelope_texts(o)]
    return texts or [stdout]


def _row(line: str) -> tuple[str, str] | None:
    """One text-form list_projects row: `  <name> <root> <branch>`; the root is quoted only
    when it contains spaces. Header/footer lines (`projects: 2 ...`, `total: 2`) are skipped."""
    if not line[:1].isspace():
        return None
    name, _, rest = line.strip().partition(" ")
    rest = rest.lstrip()
    if not name or name.endswith(":") or not rest:
        return None
    if rest.startswith('"'):
        end = rest.find('"', 1)
        root = rest[1:end] if end > 1 else ""
    else:
        root = rest.split()[0]
    return (name, root) if _ABS_PATH.match(root) else None


def parse_list_projects(stdout: str) -> dict[str, str]:
    """`list_projects` output -> {project name: root path}. Accepts the JSON form
    ({"projects": [{"name", "root_path", ...}]}, bare or inside an MCP envelope) and the
    text form."""
    out: dict[str, str] = {}
    for obj in json_objects(stdout):
        projects = obj.get("projects")
        if isinstance(projects, list):
            for p in projects:
                if isinstance(p, dict) and isinstance(p.get("name"), str) and isinstance(p.get("root_path"), str):
                    if p["name"] and p["root_path"]:
                        out[p["name"]] = p["root_path"]
    if out:
        return out
    for text in _texts(stdout):
        for line in text.splitlines():
            row = _row(line)
            if row:
                out[row[0]] = row[1]
    return out


async def list_projects(executable: str) -> dict[str, str]:
    """{project name: root path} of every indexed project (JSON form, all pages)."""
    out: dict[str, str] = {}
    offset = 0
    for _ in range(_LIST_MAX_PAGES):
        args = json.dumps({"format": "json", "limit": _LIST_PAGE, "offset": offset})
        r = await ProcessRunner().run([executable, "cli", "--quiet", "list_projects", args], timeout=30)
        if r.exit_code != 0 and offset == 0:  # a version without these arguments: plain call
            r = await ProcessRunner().run([executable, "cli", "--quiet", "list_projects"], timeout=30)
            return parse_list_projects(r.stdout)
        page = parse_list_projects(r.stdout)
        out.update(page)
        meta = next((o for o in json_objects(r.stdout) if isinstance(o.get("projects"), list)), None)
        if not page or not meta or meta.get("has_more") is not True:
            break
        offset += len(page)
    return out


def project_for_root(projects: dict[str, str], repo_path: str) -> str:
    """The project whose root is exactly this repository (never guessed from the folder name)."""
    target = _norm(repo_path)
    return next((name for name, root in projects.items() if _norm(root) == target), "")


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _first_str(obj: dict[str, Any], *keys: str) -> str:
    return next((obj[k] for k in keys if isinstance(obj.get(k), str) and obj[k]), "")


def parse_index_result(exit_code: int | None, stdout: str, stderr: str = "") -> IndexResult:
    """Interpret `cli index_repository` output. Success requires exit code 0 AND a result
    object whose status means the graph is usable (e.g. {"status": "indexed", ...}). The
    object may be bare, inside an MCP envelope, or surrounded by other protocol/log text."""
    if exit_code != 0:
        return IndexResult(False, detail=f"index_repository exit {exit_code}: {_snippet(stderr or stdout)}")
    snippet = _snippet(stdout or stderr)
    objects = json_objects(stdout)
    if any(o.get("isError") is True for o in objects if isinstance(o.get("content"), list)):
        return IndexResult(False, detail=f"index_repository returned an error: {snippet}")
    result = next((o for o in reversed(objects) if isinstance(o.get("status"), str)), None)
    if result is None:
        return IndexResult(False, detail=f"index_repository exit 0 without a recognisable result: {snippet}",
                           unrecognised=True)
    status = result["status"].strip().lower()
    res = IndexResult(status in INDEX_OK_STATUSES, status, _first_str(result, "project", "name"),
                      _first_str(result, "root_path", "repo_path"), _int(result.get("nodes")),
                      _int(result.get("edges")))
    if not res.ok:
        err = result.get("error") or result.get("message")
        res.detail = f"index_repository status={status}: {_snippet(err) if isinstance(err, str) else snippet}"
    return res


async def probe(executable: str, repo_path: str | None) -> CodebaseMemoryStatus:
    if not executable or not os.path.isfile(executable):
        return CodebaseMemoryStatus(False, detail="executable not configured/found")
    try:
        v = await ProcessRunner().run([executable, "--version"], timeout=20)
    except Exception as exc:
        return CodebaseMemoryStatus(False, detail=str(exc))
    if v.exit_code != 0:
        return CodebaseMemoryStatus(False, detail=f"--version exit {v.exit_code}")
    status = CodebaseMemoryStatus(True, version=v.stdout.strip().splitlines()[0] if v.stdout.strip() else "")
    if repo_path:
        try:
            status.project = project_for_root(await list_projects(executable), repo_path)
            status.repo_indexed = bool(status.project)
        except Exception as exc:
            status.detail = f"list_projects failed: {exc}"
    return status


async def ensure_indexed(executable: str, repo_path: str, timeout: float = 900) -> tuple[str, str]:
    """Returns (exact project name or "", detail). Indexes the repository by its absolute
    path when no project has this exact root yet. On success the detail is ALREADY_INDEXED
    or a short description of the new index. The name always comes from codebase-memory
    (list_projects matched by exact root, else the index result), never from the folder name."""
    if not executable or not os.path.isfile(executable):
        return "", "codebase-memory executable not configured"
    name = project_for_root(await list_projects(executable), repo_path)
    if name:
        return name, ALREADY_INDEXED
    root = os.path.abspath(repo_path).replace("\\", "/")
    r = await ProcessRunner().run([executable, "cli", "--quiet", "index_repository", json.dumps({"repo_path": root})],
                                  timeout=timeout)
    if r.timed_out:
        return "", f"index_repository timed out after {timeout:.0f}s"
    res = parse_index_result(r.exit_code, r.stdout, r.stderr)
    if not res.ok and not res.unrecognised:
        return "", res.detail
    # Resolve the exact project id; an unrecognised response only counts when the root is now listed.
    projects = await list_projects(executable)
    name = project_for_root(projects, repo_path)
    if not name and res.ok and res.project:
        same_root = _norm(res.root_path) == _norm(repo_path) if res.root_path else res.project not in projects
        name = res.project if same_root else ""
    if name:
        return name, res.stats() or "indexed now"
    if res.ok:
        return "", f"index_repository reported status={res.status} but no project has the root {root}"
    return "", res.detail
