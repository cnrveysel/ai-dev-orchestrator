"""Build-log reduction: compact, structured diagnostics instead of a 50,000-line log.

Generic (compiler / header-tool / linker / build-tool output of any toolchain that prints the usual
formats), so it is testable with fixture logs and usable by any provider:

    MSVC       path\\file.cpp(12,5): error C2065: 'x': undeclared identifier
    clang/gcc  path/file.cpp:12:5: error: use of undeclared identifier 'x'
    UHT        path\\File.h(12): Error: Unrecognized type 'FFoo' ...      (Unreal header tool, part of the build)
    linker     Foo.obj : error LNK2019: unresolved external symbol ...
    C#         path\\X.cs(3,1): error CS0103: ...
    build tool ERROR: ..., "Result: Failed", "BUILD FAILED"

The full log stays an artifact; prompts receive only the reduced lines (root errors first, repeats
counted, absolute paths shortened).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_MSVC = re.compile(r"^\s*(?P<file>(?:[A-Za-z]:)?[^:(\n]+?)\((?P<line>\d+)(?:,\d+)?\)\s*:\s*"
                   r"(?P<sev>fatal error|error|warning)\s+(?P<code>[A-Z]{1,4}\d{2,5})\s*:\s*(?P<msg>.+)$")
_UHT = re.compile(r"^\s*(?:LogCompile:\s*)?(?P<file>(?:[A-Za-z]:)?[^:(\n]+?\.(?:h|hpp|inl))\((?P<line>\d+)\)\s*:\s*"
                  r"(?P<sev>Error|Warning)\s*:\s*(?P<msg>.+)$")
_GCC = re.compile(r"^\s*(?P<file>(?:[A-Za-z]:)?[^:\n]+?):(?P<line>\d+):(?:\d+:)?\s*(?P<sev>fatal error|error|warning):"
                  r"\s*(?P<msg>.+)$")
_LINK = re.compile(r"^\s*(?P<file>[^:\n]*?)\s*:\s*(?P<sev>fatal error|error|warning)\s+(?P<code>LNK\d{4})\s*:\s*"
                   r"(?P<msg>.+)$")
_UHT_FAILED = re.compile(r"(UnrealHeaderTool|UHT)[^\n]*(failed|error)", re.IGNORECASE)
_TOOL_ERROR = re.compile(r"^\s*(?:ERROR|Error|error)\s*:\s*(?P<msg>.+)$")
_RESULT = re.compile(r"\bResult:\s*(?P<r>Succeeded|Failed)\b|\bBUILD (?P<b>SUCCESSFUL|FAILED)\b", re.IGNORECASE)
_BUILDING = re.compile(r"^\s*Building\s+(?P<t>[A-Za-z0-9_]+)(?:\s+and\s+[A-Za-z0-9_, ]+)?\s*\.\.\.", re.MULTILINE)
_TARGET_ARG = re.compile(r"\b(?P<t>[A-Za-z0-9_]+)\s+(?P<p>Win64|Linux|Mac|Android|IOS)\s+"
                         r"(?P<c>Debug|DebugGame|Development|Test|Shipping)\b")


@dataclass
class Diagnostic:
    category: str  # uht | compiler | linker | csharp | build_tool
    severity: str  # error | warning
    message: str
    file: str = ""
    line: int = 0
    code: str = ""
    count: int = 1

    def line_text(self) -> str:
        where = f"{self.file}({self.line})" if self.file and self.line else self.file
        what = {"uht": "UHT", "compiler": "compiler", "linker": "linker", "csharp": "C#", "build_tool": "build"}.get(
            self.category, self.category)
        return (f"{what} {self.severity}" + (f" {self.code}" if self.code else "") + (f" {where}" if where else "")
                + f": {self.message}" + (f" (x{self.count})" if self.count > 1 else ""))


@dataclass
class DiagnosticSummary:
    result: str = "unknown"  # succeeded | failed | unknown
    target: str = ""
    platform: str = ""
    configuration: str = ""
    errors: list[Diagnostic] = field(default_factory=list)  # distinct, first-seen order (roots first)
    error_count: int = 0  # all error occurrences, repeats included
    warning_count: int = 0
    uht_failed: bool = False
    lines_scanned: int = 0

    def lines(self, n: int = 8) -> list[str]:
        head = []
        what = " ".join(x for x in (self.target, self.platform, self.configuration) if x)
        head.append(f"build {self.result}" + (f" ({what})" if what else "")
                    + f": {self.error_count} error(s), {len(self.errors)} distinct, {self.warning_count} warning(s)"
                    + ("; the header tool (UHT) failed" if self.uht_failed else ""))
        return head + [d.line_text() for d in self.errors[:n]] + (
            [f"... {len(self.errors) - n} more distinct error(s) in the full log"] if len(self.errors) > n else [])

    def to_dict(self, n: int = 12) -> dict[str, Any]:
        d = asdict(self)
        d["errors"] = [asdict(e) for e in self.errors[:n]]
        return d


def _short_path(path: str, root: str) -> str:
    p = path.strip().strip('"').replace("\\", "/")
    r = root.replace("\\", "/").rstrip("/") if root else ""
    if r and p.lower().startswith(r.lower() + "/"):
        return p[len(r) + 1:]
    parts = [x for x in p.split("/") if x]
    return "/".join(parts[-3:]) if len(parts) > 3 else p


def _norm_msg(msg: str) -> str:
    return re.sub(r"\s+", " ", msg.strip())[:300]


class BuildLogReducer:
    """Streaming reducer: feed() every line as it is produced (so root errors are kept even when only the
    tail of a huge log is retained), then summary()."""

    MAX_TRACKED = 200  # distinct errors remembered while streaming (the summary shows the first few)

    def __init__(self, *, root: str = "", target: str = "", platform: str = "", configuration: str = ""):
        self.root = root
        self.s = DiagnosticSummary(target=target, platform=platform, configuration=configuration)
        self._seen: dict[tuple[str, str, int, str], Diagnostic] = {}
        self._uht: list[Diagnostic] = []
        self._other: list[Diagnostic] = []

    def _parse(self, line: str) -> Diagnostic | None:
        root = self.root
        if (m := _UHT.match(line)):
            return Diagnostic("uht", m.group("sev").lower(), _norm_msg(m.group("msg")),
                              _short_path(m.group("file"), root), int(m.group("line")))
        if (m := _MSVC.match(line)):
            code = m.group("code")
            cat = "csharp" if code.startswith("CS") else ("linker" if code.startswith("LNK") else "compiler")
            return Diagnostic(cat, "error" if "error" in m.group("sev") else "warning", _norm_msg(m.group("msg")),
                              _short_path(m.group("file"), root), int(m.group("line")), code)
        if (m := _LINK.match(line)):
            return Diagnostic("linker", "error" if "error" in m.group("sev") else "warning", _norm_msg(m.group("msg")),
                              _short_path(m.group("file"), root), 0, m.group("code"))
        if (m := _GCC.match(line)):
            return Diagnostic("compiler", "error" if "error" in m.group("sev") else "warning",
                              _norm_msg(m.group("msg")), _short_path(m.group("file"), root), int(m.group("line")))
        if (m := _TOOL_ERROR.match(line)):
            return Diagnostic("build_tool", "error", _norm_msg(m.group("msg")))
        return None

    def feed(self, raw: str) -> None:
        s = self.s
        line = _ANSI.sub("", raw).rstrip()
        if not line.strip():
            return
        s.lines_scanned += 1
        if _UHT_FAILED.search(line):
            s.uht_failed = True
        if (m := _RESULT.search(line)):
            r = (m.group("r") or m.group("b") or "").lower()
            s.result = "succeeded" if r in ("succeeded", "successful") else "failed"
        if not s.target and (m := _BUILDING.match(line)):
            s.target = m.group("t")
        if (not s.platform or not s.configuration) and (m := _TARGET_ARG.search(line)):
            s.target = s.target or m.group("t")
            s.platform = s.platform or m.group("p")
            s.configuration = s.configuration or m.group("c")
        d = self._parse(line)
        if d is None:
            return
        if d.severity == "warning":
            s.warning_count += 1
            return
        s.error_count += 1
        key = (d.category, d.file.lower(), d.line, d.message.lower())
        if key in self._seen:
            self._seen[key].count += 1
            return
        if len(self._seen) >= self.MAX_TRACKED:
            return
        self._seen[key] = d
        (self._uht if d.category == "uht" else self._other).append(d)

    def summary(self, max_distinct: int = 12) -> DiagnosticSummary:
        s = self.s
        if self._uht:
            s.uht_failed = True
        # Header-tool errors come first: code generation failing makes every later compile error a consequence.
        s.errors = (self._uht + self._other)[:max(1, max_distinct)]
        if s.result == "unknown" and s.error_count:
            s.result = "failed"
        return s


def reduce_build_log(text: str, *, root: str = "", target: str = "", platform: str = "", configuration: str = "",
                     max_distinct: int = 12) -> DiagnosticSummary:
    """Reduce a build log to its root errors (deduplicated, counted, compact)."""
    r = BuildLogReducer(root=root, target=target, platform=platform, configuration=configuration)
    for line in (text or "").splitlines():
        r.feed(line)
    return r.summary(max_distinct)
