"""Unreal Engine evidence provider. Unreal awareness lives ONLY here: the scheduler, the evidence models and
the checkpoint semantics know nothing about Unreal - they see generic capabilities and results.

AI Studio's Unreal philosophy (this provider verifies it, it never decides it):
  - gameplay systems are implemented in C++; Blueprint graphs are never authored automatically
  - designer-facing properties/functions may be exposed intentionally: Blueprint exposure is an API
    boundary, not the implementation language; internal/transient/lifecycle machinery stays private
The provider checks builds (UnrealBuildTool - the header tool runs as PART of the build, so there is no
separate "UHT command"; the build's diagnostics tell header-tool errors from compiler errors), automation
tests, declared read-only commandlets and a running game/editor (log capture, crash detection). It never
authors or edits Blueprints, never resaves assets and makes no architectural or gameplay decisions.

Discovery is bounded and read-only:
  project  the .uproject in the repository root (no .uproject -> no Unreal capability at all)
  engine   provider setting unreal.engine_root, else the project's EngineAssociation (Epic launcher
           installs and registered source builds: read-only registry / launcher manifest), else an engine
           tree enclosing the project; always validated by Engine/Build/Build.version
  tools    UnrealBuildTool, UnrealEditor / UnrealEditor-Cmd (UE5) or UE4Editor / UE4Editor-Cmd (UE4)
Whatever cannot be found is reported UNAVAILABLE with the reason - nothing is assumed.

Provider settings (Settings -> Orchestration -> evidence provider settings): unreal.engine_root,
unreal.test_filter (default automation filter), unreal.ready_marker / unreal.editor_ready_marker (log text
that means the game / editor finished starting), unreal.configuration (default Development).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.evidence import readiness as rd
from app.evidence.diagnostics import BuildLogReducer
from app.evidence.lifecycle import ServiceProcess
from app.evidence.freshlog import FreshLog
from app.evidence.model import ERROR, FAILED, INFERRED, PASSED, TIMEOUT, UNAVAILABLE, Capability, PlanStep
from app.evidence.providers.base import CapabilityProvider, DiscoveryContext, ExecContext, StepOutcome
from app.evidence.providers.command import MANIFEST, declared_provenance, run_command

PROJECT_MAX_BYTES = 256 * 1024
LOG_TAIL_BYTES = 64 * 1024
_TEST_FILTER = re.compile(r"^[A-Za-z0-9_.+-]{1,120}$")
_COMMANDLET = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,60}$")
_ARG = re.compile(r"^-[A-Za-z0-9_]{1,40}(=[A-Za-z0-9_.+:/-]{1,120})?$")
# Commandlets that save, move or rewrite assets: never offered (verification must not change the project).
MUTATING_COMMANDLETS = frozenset(x.lower() for x in (
    "ResavePackages", "FixupRedirects", "ResaveAssets", "WrangleContent", "ReplaceActor", "SwapSoundForDialogue",
    "GenerateDistillFileSets", "MigrateContent", "DeleteRedirectors", "UpdateGameProjectCommandlet"))
FATAL_MARKERS = ("Fatal error!", "Unhandled Exception:", "Assertion failed:", "=== Critical error: ===",
                 "LogWindows: Error: === Critical error")
LIVE_CODING_MARKERS = ("Unable to build while Live Coding is active", "Live Coding is active")
DEFAULT_GAME_READY = "LogLoad: Took"  # the game world finished loading its map
DEFAULT_EDITOR_READY = "LogLoad: (Engine Initialization) Total time:"
_RESULT_LINE = re.compile(r"Test Completed\. Result=\{(?P<r>\w+)\}\s*Name=\{(?P<n>[^}]*)\}(?:\s*Path=\{(?P<p>[^}]*)\})?")
_FOUND_LINE = re.compile(r"Found (?P<n>\d+) automation tests? based on '(?P<f>[^']*)'")
_TEST_EXIT_LINE = re.compile(r"TEST COMPLETE\. EXIT CODE: (?P<c>-?\d+)")
SUCCESS_STATES = frozenset({"success", "succeeded", "successwithwarnings", "passed"})
AUTOMATION_FLAGS = ("-unattended", "-nop4", "-nosplash", "-nopause", "-NullRHI", "-NoSound", "-stdout",
                    "-FullStdOutLogOutput")


def _platform() -> str:
    return {"win32": "Win64", "darwin": "Mac"}.get(sys.platform, "Linux")


def _exe(name: str) -> str:
    return name + (".exe" if sys.platform == "win32" else "")


@dataclass
class UnrealProject:
    path: str  # absolute path of the .uproject
    name: str
    association: str = ""
    modules: list[dict[str, str]] = field(default_factory=list)  # Name, Type
    targets: list[str] = field(default_factory=list)  # from Source/*.Target.cs
    plugins: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def editor_target(self) -> str:
        if f"{self.name}Editor" in self.targets:
            return f"{self.name}Editor"
        return next((t for t in self.targets if t.endswith("Editor")), self.targets[0] if self.targets else "")


@dataclass
class UnrealEngine:
    root: str
    version: str = ""
    how: str = ""  # how it was found
    ubt: str = ""
    build_script: str = ""
    editor: str = ""
    editor_cmd: str = ""


def read_project(repo: str) -> UnrealProject | None:
    try:
        found = sorted(n for n in os.listdir(repo) if n.lower().endswith(".uproject"))
    except OSError:
        return None
    if not found:
        return None
    path = os.path.join(os.path.abspath(repo), found[0])
    p = UnrealProject(path, os.path.splitext(found[0])[0])
    try:
        if os.path.getsize(path) > PROJECT_MAX_BYTES:
            raise ValueError("larger than 256 KB")
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        p.error = f"the project descriptor is unreadable: {exc}"[:200]
        return p
    p.association = str(data.get("EngineAssociation") or "")[:80]
    p.modules = [{"Name": str(m.get("Name") or ""), "Type": str(m.get("Type") or "")}
                 for m in data.get("Modules") or [] if isinstance(m, dict) and m.get("Name")][:40]
    p.plugins = [str(x.get("Name")) for x in data.get("Plugins") or [] if isinstance(x, dict) and x.get("Name")][:80]
    src = os.path.join(os.path.dirname(path), "Source")
    try:
        p.targets = sorted(n[:-len(".Target.cs")] for n in os.listdir(src) if n.endswith(".Target.cs"))[:20]
    except OSError:
        p.targets = []
    return p


def engine_from_root(root: str, how: str = "") -> UnrealEngine | None:
    """An engine tree, validated by Engine/Build/Build.version, with the tools that actually exist."""
    if not root or "\0" in root:
        return None
    root = os.path.abspath(root)
    ver_file = os.path.join(root, "Engine", "Build", "Build.version")
    if not os.path.isfile(ver_file):
        return None
    version = ""
    try:
        with open(ver_file, encoding="utf-8-sig") as f:
            v = json.load(f)
        version = f"{v.get('MajorVersion')}.{v.get('MinorVersion')}.{v.get('PatchVersion', 0)}"
    except (OSError, ValueError):
        version = "unknown"
    e = UnrealEngine(root, version, how)
    plat = {"Win64": "Win64", "Mac": "Mac"}.get(_platform(), "Linux")
    binaries = os.path.join(root, "Engine", "Binaries")
    for cand in (os.path.join(binaries, "DotNET", "UnrealBuildTool", _exe("UnrealBuildTool")),
                 os.path.join(binaries, "DotNET", _exe("UnrealBuildTool"))):
        if os.path.isfile(cand):
            e.ubt = cand
            break
    script = os.path.join(root, "Engine", "Build", "BatchFiles", "Build.bat" if plat == "Win64" else
                          os.path.join(plat, "Build.sh"))
    e.build_script = script if os.path.isfile(script) else ""
    for editor, cmd in (("UnrealEditor", "UnrealEditor-Cmd"), ("UE4Editor", "UE4Editor-Cmd")):
        ed, ec = os.path.join(binaries, plat, _exe(editor)), os.path.join(binaries, plat, _exe(cmd))
        if os.path.isfile(ed) or os.path.isfile(ec):
            e.editor = ed if os.path.isfile(ed) else ""
            e.editor_cmd = ec if os.path.isfile(ec) else ""
            break
    return e


def _registry_engine(association: str) -> str:
    try:
        import winreg  # read-only lookups; absent off Windows
    except ImportError:
        return ""
    try:
        if association.startswith("{"):  # a registered source build
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Epic Games\Unreal Engine\Builds") as k:
                return str(winreg.QueryValueEx(k, association)[0])
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\EpicGames\Unreal Engine\{association}") as k:
            return str(winreg.QueryValueEx(k, "InstalledDirectory")[0])
    except OSError:
        return ""


def _launcher_engine(association: str) -> str:
    base = os.environ.get("ProgramData") or r"C:\ProgramData"
    path = os.path.join(base, "Epic", "UnrealEngineLauncher", "LauncherInstalled.dat")
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > 2 * 1024 * 1024:
            return ""
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return ""
    for item in data.get("InstallationList") or [] if isinstance(data, dict) else []:
        if isinstance(item, dict) and str(item.get("AppName") or "") == f"UE_{association}":
            return str(item.get("InstallLocation") or "")
    return ""


def find_engine(project: UnrealProject, repo: str, configured: str = "") -> tuple[UnrealEngine | None, str]:
    """The default engine locator (read-only). Returns (engine, how) or (None, why not)."""
    if configured:
        e = engine_from_root(configured, "provider setting unreal.engine_root")
        return (e, e.how) if e else (None, f"unreal.engine_root {configured!r} is not an Unreal Engine directory")
    a = project.association
    if a:
        for how, root in (("Epic launcher / registry", _registry_engine(a)), ("Epic launcher manifest",
                                                                              _launcher_engine(a))):
            e = engine_from_root(root, f"{how} ({a})") if root else None
            if e:
                return e, e.how
    d = os.path.dirname(os.path.abspath(repo))
    for _ in range(4):  # a project inside an engine source tree (no association)
        e = engine_from_root(d, "engine tree enclosing the project")
        if e:
            return e, e.how
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None, (f"Unreal Engine for EngineAssociation {a!r} not found" if a else "no Unreal Engine found") + \
        " - set the evidence provider setting unreal.engine_root"


def build_argv(e: UnrealEngine, p: UnrealProject, target: str, platform: str, config: str) -> list[str] | str:
    """The UnrealBuildTool invocation (the header tool runs as part of it). An argv list - no shell - when the
    UnrealBuildTool executable exists; otherwise the engine's build script as one quoted command line."""
    args = [target, platform, config, f"-Project={p.path}", "-WaitMutex", "-NoHotReloadFromIDE"]
    if e.ubt:
        return [e.ubt, *args]
    return " ".join([f'"{e.build_script}"', target, platform, config, f'-Project="{p.path}"', "-WaitMutex",
                     "-NoHotReloadFromIDE"])


def _quotable(path: str) -> bool:
    return bool(path) and not any(c in path for c in '"%^&|<>\r\n')


class UnrealProvider(CapabilityProvider):
    name = "unreal"
    version = "2"  # 2: automation decides from matching per-test results of this run only (stdout/report/fresh log)

    def __init__(self, locator: Callable[[UnrealProject, str, str], tuple[UnrealEngine | None, str]] | None = None,
                 tool_prefix: Callable[[str], list[str]] | None = None):
        self.locator = locator or find_engine
        # How a tool path is started (tests map it to a deterministic fake; production: the path itself).
        self.tool_prefix = tool_prefix or (lambda path: [path])
        self.last_notes: list[str] = []

    # ------------------------------------------------------------ discovery

    async def discover(self, ctx: DiscoveryContext) -> list[Capability]:
        self.last_notes = []
        p = read_project(ctx.repo_path)
        if p is None:
            return []  # not an Unreal project: no Unreal capability at all
        key = f"unreal:{p.name}"
        prov = {"source_detail": "Unreal project", "declared_in": os.path.basename(p.path)}

        def cap(suffix: str, kind: str, name: str, desc: str, reason: str, *, ok: bool,
                constraints: dict[str, Any] | None = None, details: dict[str, Any] | None = None) -> Capability:
            return Capability(id="", kind=kind, name=name[:120], description=desc[:300], source="environment",
                              confidence=INFERRED if ok else UNAVAILABLE, provider=self.name, key=f"{key}:{suffix}",
                              constraints={"timeout_seconds": 600, "accepts_args": False, **(constraints or {})},
                              details={"project": p.path, "project_name": p.name, **(details or {})}, reason=reason,
                              provenance=dict(prov))

        caps = [cap("project", "static_analysis", f"Unreal project descriptor: {p.name}",
                    "checks that the .uproject parses and that its C++ modules and targets exist (static only)",
                    p.error or "Unreal project found; descriptor readable", ok=not p.error)]
        if p.error:
            return caps
        e, how = self.locator(p, ctx.repo_path, ctx.provider_setting("unreal.engine_root"))
        platform = _platform()
        config = ctx.provider_setting("unreal.configuration", "Development")
        if config not in ("Debug", "DebugGame", "Development", "Test", "Shipping"):
            config = "Development"
        eng = {"engine_root": e.root, "engine_version": e.version, "platform": platform} if e else {}
        if e is not None:
            prov["engine"] = f"{e.version} ({e.how})"
        no_engine = f"Unreal Engine unavailable: {how}"
        target = p.editor_target
        if e is None:
            why = no_engine
        elif not p.targets:
            why = "no C++ targets (Source/*.Target.cs): nothing for UnrealBuildTool to build"
        elif not e.ubt and not (e.build_script and _quotable(e.build_script) and _quotable(p.path)):
            why = "UnrealBuildTool not found in the engine"
        else:
            why = ""
        caps.append(cap("build", "build", f"Unreal build: {target or p.name} {platform} {config}",
                        "UnrealBuildTool build of the project (runs the header tool as part of the build; the result "
                        "separates header-tool and compiler errors)",
                        why or f"engine {e.version} ({e.how}); not run yet (a build may need a larger evidence step "
                               "timeout)", ok=not why,
                        constraints={"timeout_seconds": 3600},
                        details={**eng, "target": target, "configuration": config}))
        caps.append(cap("uht", "header_validation", "Unreal header tool (standalone)",
                        "the header tool is not run on its own",
                        "UnrealHeaderTool runs as part of the build: use the build capability (its diagnostics "
                        "separate header-tool errors)", ok=False))
        ecmd_ok = e is not None and bool(e.editor_cmd)
        default_filter = ctx.provider_setting("unreal.test_filter")
        if default_filter and not _TEST_FILTER.match(default_filter):
            default_filter = ""
        caps.append(cap("automation", "automated_test", "Unreal automation tests",
                        "runs the automation tests matching a filter (argument; e.g. the project's test prefix) "
                        "headless (-NullRHI) and reads the per-test results of THIS run (stdout, report, its own "
                        "log): passes only when at least one test under the filter completed and all of them "
                        "succeeded - an exit code alone never passes",
                        no_engine if e is None else ("" if ecmd_ok else "UnrealEditor-Cmd not found in the engine")
                        or "engine found; not run yet", ok=ecmd_ok,
                        constraints={"timeout_seconds": 1800, "accepts_args": True, "max_args": 1,
                                     "args_help": "the automation test filter"
                                                  + (f" (default {default_filter})" if default_filter else "")},
                        details={**eng, "arg_pattern": _TEST_FILTER.pattern, "default_filter": default_filter}))
        caps += self._commandlets(ctx, p, e, eng, cap)
        ed_ok = e is not None and bool(e.editor)
        ed_why = no_engine if e is None else ("" if ed_ok else "UnrealEditor not found in the engine")
        game_ready = ctx.provider_setting("unreal.ready_marker", DEFAULT_GAME_READY)[:200]
        editor_ready = ctx.provider_setting("unreal.editor_ready_marker", DEFAULT_EDITOR_READY)[:200]
        for suffix, kind, label, mode, marker, ready in (
                ("game", "runtime_launch", "Unreal game (standalone -game)", "game", game_ready, 300),
                ("editor", "editor_launch", "Unreal editor", "editor", editor_ready, 900)):
            spec, _bad = rd.parse_readiness({"kind": "stdout_contains", "text": marker, "timeout_seconds": ready})
            caps.append(cap(suffix, kind, f"{label}: {p.name}",
                            f"starts the {label.lower()} and waits until its log shows {marker!r}",
                            ed_why or "engine found; started only while evidence is collected", ok=ed_ok,
                            constraints={"lifecycle": "service", "readiness": spec, "ready_timeout_seconds": ready,
                                         "timeout_seconds": 60},
                            details={**eng, "mode": mode}))
        launch_ok = ed_ok
        caps.append(cap("log", "runtime_log", f"Unreal log capture: {p.name}",
                        "bounded capture of the running game/editor output and Saved/Logs; use expect_output / "
                        "fail_output as literal markers", ed_why or "observes a running Unreal launch",
                        ok=launch_ok, constraints={"requires_resource": "*", "timeout_seconds": 30},
                        details={"log": f"Saved/Logs/{p.name}.log"}))
        caps.append(cap("crash", "crash_detection", f"Unreal crash detection: {p.name}",
                        "observes the running game/editor for N seconds (argument, default 5) and passes if it stays "
                        "alive with no fatal error in its log and no new crash report", ed_why or
                        "observes a running Unreal launch", ok=launch_ok,
                        constraints={"requires_resource": "*", "timeout_seconds": 330, "accepts_args": True,
                                     "max_args": 1, "args_help": "seconds to observe (1-300)"},
                        details={"arg_pattern": r"[0-9]{1,3}"}))
        caps.append(cap("performance", "performance", "Unreal performance capture",
                        "frame-time capture", "not implemented: no truthful performance capture is available",
                        ok=False))
        return caps

    def _commandlets(self, ctx: DiscoveryContext, p: UnrealProject, e: UnrealEngine | None, eng: dict[str, Any],
                     cap: Callable[..., Capability]) -> list[Capability]:
        """Only commandlets the project explicitly declares (.aistudio/evidence.json -> unreal.commandlets);
        commandlets that save or rewrite assets are never offered."""
        path = os.path.join(ctx.repo_path, *MANIFEST.split("/"))
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f) if os.path.getsize(path) <= 64 * 1024 else {}
        except (OSError, ValueError):
            return []
        decl = ((data or {}).get("unreal") or {}).get("commandlets") if isinstance(data, dict) else None
        out = []
        for d in (decl if isinstance(decl, list) else [])[:10]:
            if not isinstance(d, dict):
                continue
            name, cmdlet = str(d.get("name") or d.get("commandlet") or "")[:60], str(d.get("commandlet") or "")
            args = [str(a) for a in d.get("args") or []][:8]
            why = ""
            if not _COMMANDLET.match(cmdlet):
                why = f"invalid commandlet name {cmdlet!r}"
            elif cmdlet.lower() in MUTATING_COMMANDLETS:
                why = f"{cmdlet} saves or rewrites assets: never used for verification"
            elif not all(_ARG.match(a) for a in args):
                why = "commandlet arguments must look like -Name or -Name=value (no spaces or quotes)"
            elif e is None or not e.editor_cmd:
                why = "UnrealEditor-Cmd not found" if e is not None else "Unreal Engine unavailable"
            kind = re.sub(r"[^a-z0-9]+", "_", str(d.get("kind") or "static_analysis").lower()).strip("_")
            c = cap(f"commandlet:{cmdlet.lower()}", kind or "static_analysis", f"Unreal commandlet: {name}",
                    str(d.get("description") or f"runs -run={cmdlet} headless"), why or "declared; not run yet",
                    ok=not why, constraints={"timeout_seconds": int(d.get("timeout_seconds") or 1800)},
                    details={**eng, "commandlet": cmdlet, "args": args})
            c.provenance.update(declared_provenance(ctx, MANIFEST))
            out.append(c)
        return out

    # ------------------------------------------------------------ validation / execution

    def validate(self, cap: Capability, step: PlanStep) -> str:
        if cap.constraints.get("lifecycle") == "service":
            return f"{cap.id} is a launch: list it under `resources` (it is started and stopped around the steps)"
        if cap.kind == "automated_test" and cap.key.endswith(":automation"):
            if len(step.args) > 1:
                return f"{cap.id} takes one argument: the automation test filter"
            flt = step.args[0] if step.args else str(cap.details.get("default_filter") or "")
            if not flt:
                return f"{cap.id} needs the automation test filter as its argument (e.g. the project's test prefix)"
            if not _TEST_FILTER.match(flt):
                return f"automation filter {flt!r} must match {_TEST_FILTER.pattern}"
            return ""
        if cap.kind == "crash_detection":
            if len(step.args) > 1 or (step.args and not (step.args[0].isdigit() and 1 <= int(step.args[0]) <= 300)):
                return f"{cap.id} takes one optional argument: the seconds to observe (1-300)"
            return ""
        return "" if not step.args else f"{cap.id} does not accept arguments"

    def _tool(self, path: str) -> list[str]:
        return list(self.tool_prefix(path))

    async def execute(self, cap: Capability, step: PlanStep, ctx: ExecContext) -> StepOutcome:
        why = self.validate(cap, step)
        if why:
            return StepOutcome(UNAVAILABLE, reason=why, cause="strategy")
        suffix = cap.key.split(":", 2)[-1]
        p = read_project(ctx.repo_path)
        if p is None or p.error:
            return StepOutcome(UNAVAILABLE, reason="the Unreal project descriptor is not readable", cause="strategy")
        if suffix == "project":
            return self._check_project(p, ctx)
        if suffix in ("log", "crash"):
            return await self._observe(cap, step, suffix, p, ctx)
        e = engine_from_root(str(cap.details.get("engine_root") or ""), "discovered")
        if e is None:
            return StepOutcome(UNAVAILABLE, reason="the Unreal Engine found at discovery is not available any more",
                               cause="strategy")
        if suffix == "build":
            return await self._build(cap, step, ctx, e, p)
        if suffix == "automation":
            return await self._automation(cap, step, ctx, e, p)
        if suffix.startswith("commandlet:"):
            cmdlet, args = str(cap.details.get("commandlet")), list(cap.details.get("args") or [])
            argv = self._tool(e.editor_cmd) + [p.path, f"-run={cmdlet}", *args, "-unattended", "-nop4", "-nosplash",
                                               "-NullRHI", "-stdout", "-FullStdOutLogOutput"]
            return await run_command(argv, step, ctx)
        return StepOutcome(UNAVAILABLE, reason=f"unknown Unreal capability {suffix}", cause="strategy")

    def _check_project(self, p: UnrealProject, ctx: ExecContext) -> StepOutcome:
        problems = []
        root = os.path.dirname(p.path)
        for m in p.modules:
            build_cs = os.path.join(root, "Source", m["Name"], f"{m['Name']}.Build.cs")
            if not os.path.isfile(build_cs):
                problems.append(f"module {m['Name']}: Source/{m['Name']}/{m['Name']}.Build.cs is missing")
        if p.modules and not p.targets:
            problems.append("C++ modules are declared but Source/ has no *.Target.cs")
        summary = (f"{os.path.basename(p.path)}: {len(p.modules)} module(s), target(s) "
                   f"{', '.join(p.targets) or 'none'}")
        if problems:
            return StepOutcome(FAILED, summary=f"{summary}; " + "; ".join(problems[:4]), reason="descriptor problems",
                               output="\n".join(problems))
        return StepOutcome(PASSED, summary=f"{summary}; descriptor consistent (static check only)")

    async def _build(self, cap: Capability, step: PlanStep, ctx: ExecContext, e: UnrealEngine,
                     p: UnrealProject) -> StepOutcome:
        target = str(cap.details.get("target") or p.editor_target)
        platform, config = str(cap.details.get("platform") or _platform()), str(cap.details.get("configuration")
                                                                              or "Development")
        argv = build_argv(e, p, target, platform, config)
        if isinstance(argv, list):
            argv = self._tool(argv[0]) + argv[1:]
        reducer = BuildLogReducer(root=ctx.repo_path, target=target, platform=platform, configuration=config)
        out = await run_command(argv, step, ctx, reducer=reducer)
        if any(m in out.output for m in LIVE_CODING_MARKERS):  # the environment blocks the build, not the code
            return StepOutcome(UNAVAILABLE, out.exit_code, summary="Unreal build blocked: Live Coding is active in a "
                                                                   "running editor", reason="Live Coding is active",
                               output=out.output, cause="strategy", duration=out.duration)
        s = reducer.summary()
        if out.status == FAILED and s.uht_failed:
            first = next((d for d in s.errors if d.category == "uht"), None)
            out.summary = f"Unreal build of {target} failed in the header tool (UHT)" + \
                (f": {first.line_text()}" if first else "")
        elif out.status == FAILED and s.errors:
            out.summary = f"Unreal build of {target} failed: {s.errors[0].line_text()}"[:400]
        elif out.status == PASSED:
            out.summary = f"Unreal build of {target} {platform} {config} succeeded" + \
                (f" ({s.warning_count} warning(s))" if s.warning_count else "")
        return out

    async def _automation(self, cap: Capability, step: PlanStep, ctx: ExecContext, e: UnrealEngine,
                          p: UnrealProject) -> StepOutcome:
        """Runs the automation tests matching the filter and decides ONLY from results this run produced: its
        stdout, its report (in a directory created just now) and the project log (Saved/Logs/<project>.log) only
        as far as this run provably wrote it - snapshot before the launch, then only the rotated/new file or the
        appended part (FreshLog); nothing in the project is deleted or rotated by AI Studio. Exit code 0 alone is
        never a pass: at least one test matching the filter must have completed, and every one with success."""
        flt = step.args[0] if step.args else str(cap.details.get("default_filter") or "")
        work = tempfile.mkdtemp(prefix="aio-ue-automation-")  # outside the project; removed afterwards
        report = os.path.join(work, "report")
        os.makedirs(report, exist_ok=True)
        project_log = FreshLog.snapshot(os.path.join(os.path.dirname(p.path), "Saved", "Logs", f"{p.name}.log"))
        live = AutomationLog()
        tool = self._tool(e.editor_cmd)
        if not all(_quotable(x) for x in [*tool, p.path, report]):
            shutil.rmtree(work, ignore_errors=True)
            return StepOutcome(UNAVAILABLE, reason="a path of the automation run contains characters that cannot be "
                                                   "passed safely", cause="strategy")
        cmd = automation_command(tool, p.path, flt, report)
        try:  # the planner's markers are checked below against everything this run produced, not stdout alone
            out = await run_command(cmd, dataclasses.replace(step, expect_output="", fail_output=""), ctx,
                                    on_line=live.feed)
            sources: list[tuple[str, AutomationLog]] = [("stdout", live)]
            rep = AutomationLog()
            for state, path in _read_report(report):
                rep.add(state, path)
            sources.append(("report", rep))
            texts = [out.output]
            fresh_log, _how = project_log.fresh_text()  # "" when this run did not provably write it (stale)
            texts.append(fresh_log)
            sources.append(("project log", AutomationLog.parse(fresh_log)))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if out.status in (TIMEOUT, UNAVAILABLE):
            return out
        seen = AutomationLog.merge([src for _n, src in sources])
        used = [n for n, src in sources if src.results or src.found is not None]
        matched = {path: st for path, st in seen.results.items() if filter_matches(path, flt)}
        unrelated = sorted(set(seen.results) - set(matched))
        failed = sorted(path for path, st in matched.items() if st.lower() not in SUCCESS_STATES)
        text = "\n".join(t for t in texts if t)
        details = {"automation": {"filter": flt, "found": seen.found, "ran": len(matched), "failed": len(failed),
                                  "failures": failed[:10], "unrelated": unrelated[:5], "sources": used,
                                  "test_exit_code": seen.exit_code}}
        common = dict(exit_code=out.exit_code, output=out.output, duration=out.duration, details=details,
                      truncated=out.truncated)
        where = f" (observed in {', '.join(used)})" if used else ""
        if step.fail_output and step.fail_output in text:
            return StepOutcome(FAILED, summary=f"automation run ({flt}) printed the failure marker {step.fail_output!r}",
                               reason="failure marker found", **common)
        if failed:
            return StepOutcome(FAILED, summary=f"{len(failed)} of {len(matched)} automation test(s) matching {flt!r} "
                                               f"failed: {', '.join(failed[:4])}", reason="automation test failures",
                               diagnostics=[f"failed: {x}" for x in failed[:8]], **common)
        if matched:
            if out.exit_code not in (0, None):
                return StepOutcome(FAILED, summary=f"the automation run exited {out.exit_code} although the tests "
                                                   f"matching {flt!r} reported success", reason=f"exit {out.exit_code}",
                                   **common)
            if step.expect_output and step.expect_output not in text:
                return StepOutcome(FAILED, summary=f"{len(matched)} automation test(s) matching {flt!r} passed but the "
                                                   f"expected text {step.expect_output!r} did not appear",
                                   reason="expected output missing", cause="unobserved", **common)
            return StepOutcome(PASSED, summary=f"{len(matched)} automation test(s) matching {flt!r} passed{where}: "
                                               + ", ".join(sorted(matched)[:4]), **common)
        if seen.found == 0:
            return StepOutcome(FAILED, summary=f"no automation test matches {flt!r} (the run found 0 tests)",
                               reason="no test matched the filter", **common)
        if unrelated:
            return StepOutcome(FAILED, summary=f"no test matching {flt!r} completed; only unrelated test(s) reported a "
                                               f"result: {', '.join(unrelated[:3])}", reason="no matching test result",
                               **common)
        if out.exit_code not in (0, None):
            return StepOutcome(FAILED, summary=f"the automation run exited {out.exit_code} without a test result",
                               reason=f"exit {out.exit_code}", **common)
        return StepOutcome(ERROR, summary=f"the automation run ({flt}) exited 0 but no test result was observed in its "
                                          "output, its report or a log it wrote: an exit code alone is not evidence",
                           reason="no observable automation result", cause="strategy", **common)

    async def _observe(self, cap: Capability, step: PlanStep, suffix: str, p: UnrealProject,
                       ctx: ExecContext) -> StepOutcome:
        if not ctx.resources:
            return StepOutcome(UNAVAILABLE, reason="no running game/editor for this requirement", cause="strategy")
        rkey, h = next(iter(ctx.resources.items()))
        t0 = time.monotonic()
        if suffix == "crash":  # watch it run for a while: "it started" is not "it keeps running"
            window = min(float(step.args[0]) if step.args else 5.0, max(0.5, ctx.timeout - 1))
            while time.monotonic() - t0 < window and getattr(h, "running", False) and \
                    not any(m in (h.output() if hasattr(h, "output") else "") for m in FATAL_MARKERS):
                await asyncio.sleep(0.1)
        output = h.output() if hasattr(h, "output") else ""
        log_rel = f"Saved/Logs/{p.name}.log"
        log_path = os.path.join(os.path.dirname(p.path), *log_rel.split("/"))
        text = output
        if os.path.isfile(log_path):
            text += f"\n=== {log_rel} ===\n" + _tail(log_path)
        text = text[-max(4096, ctx.max_capture_bytes):]
        if suffix == "log":
            if step.fail_output and step.fail_output in text:
                return StepOutcome(FAILED, summary=f"{cap.name}: the failure marker {step.fail_output!r} appeared",
                                   reason="failure marker found", output=text)
            if step.expect_output and step.expect_output not in text:
                return StepOutcome(FAILED, summary=f"{cap.name}: the expected text {step.expect_output!r} did not "
                                                   "appear", reason="expected output missing", output=text)
            return StepOutcome(PASSED, summary=f"{cap.name}: captured ({len(text)} chars)"
                               + (f", contains {step.expect_output!r}" if step.expect_output else "")
                               + (f", no {step.fail_output!r}" if step.fail_output else ""), output=text)
        fatal = next((m for m in FATAL_MARKERS if m in text), "")
        started = getattr(h, "started_monotonic", 0.0)
        crashes = _new_crash_reports(os.path.dirname(p.path), time.time() - (time.monotonic() - started)
                                     if started else time.time())
        watched = time.monotonic() - t0
        if fatal or crashes or not getattr(h, "running", False):
            what = (f"fatal marker {fatal!r} in the log" if fatal else
                    (f"new crash report {crashes[0]}" if crashes else f"exited with code {getattr(h, 'exit_code', '?')}"))
            return StepOutcome(FAILED, summary=f"{p.name} crashed or stopped: {what}", reason="crash detected",
                               output=text, duration=watched)
        return StepOutcome(PASSED, summary=f"{p.name} ({rkey}) ran for {watched:.1f}s with no fatal error and no crash "
                                           "report", output=text, duration=watched)

    # ------------------------------------------------------------ lifecycle (game / editor launch)

    def supports_lifecycle(self, cap: Capability) -> bool:
        return cap.constraints.get("lifecycle") == "service"

    async def start(self, cap: Capability, ctx: ExecContext) -> ServiceProcess:
        p = read_project(ctx.repo_path)
        e = engine_from_root(str(cap.details.get("engine_root") or ""), "discovered")
        if p is None or e is None or not e.editor:
            raise RuntimeError("the Unreal project or editor is not available")
        mode = cap.details.get("mode")
        extra = ["-game", "-windowed", "-ResX=1280", "-ResY=720"] if mode == "game" else []
        argv = self._tool(e.editor) + [p.path, *extra, "-log", "-stdout", "-FullStdOutLogOutput", "-nosplash",
                                       "-unattended"]
        sp = ServiceProcess(argv, os.path.dirname(p.path), ctx.max_capture_bytes)
        spec = cap.constraints.get("readiness") or {}
        await sp.start([str(spec.get("text") or "")] + list(FATAL_MARKERS))
        return sp

    async def ready(self, handle: ServiceProcess, cap: Capability, ctx: ExecContext) -> tuple[str, str]:
        declared = float(cap.constraints.get("ready_timeout_seconds") or ctx.ready_timeout)
        spec = cap.constraints.get("readiness") or {"kind": "process_alive", "settle_seconds": 5}
        return await rd.wait_ready(handle, spec, min(declared, ctx.ready_timeout), ctx.repo_path)

    def capture(self, handle: ServiceProcess) -> str:
        return handle.output()

    async def stop(self, handle: ServiceProcess) -> None:
        await handle.stop()


def _tail(path: str, limit: int = LOG_TAIL_BYTES) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _new_crash_reports(project_dir: str, since: float) -> list[str]:
    d = os.path.join(project_dir, "Saved", "Crashes")
    try:
        return sorted(n for n in os.listdir(d) if os.path.getmtime(os.path.join(d, n)) >= since - 1)[:5]
    except OSError:
        return []


def _read_report(report_dir: str) -> list[tuple[str, str]]:
    """(state, test path) from the automation report (index.json), bounded; [] when absent/unreadable."""
    for dirpath, _dirs, files in os.walk(report_dir):
        if "index.json" in files:
            path = os.path.join(dirpath, "index.json")
            try:
                if os.path.getsize(path) > 5 * 1024 * 1024:
                    return []
                with open(path, encoding="utf-8-sig") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                return []
            return [(str(t.get("state") or ""), str(t.get("fullTestPath") or t.get("testDisplayName") or ""))
                    for t in (data.get("tests") or [] if isinstance(data, dict) else [])[:2000] if isinstance(t, dict)]
    return []


class AutomationLog:
    """What an automation run reported: discovery ("Found N automation tests based on 'X'"), per-test results
    ("Test Completed. Result={..} Name={..} Path={..}") and the automation exit line. Bounded."""
    MAX_RESULTS = 2000

    def __init__(self) -> None:
        self.results: dict[str, str] = {}  # full test path -> state
        self.found: int | None = None
        self.exit_code: int | None = None

    def add(self, state: str, path: str) -> None:
        path = path.strip()
        if path and (path in self.results or len(self.results) < self.MAX_RESULTS):
            self.results[path] = state

    def feed(self, line: str) -> None:
        m = _RESULT_LINE.search(line)
        if m:
            self.add(m.group("r"), m.group("p") or m.group("n"))
            return
        m = _FOUND_LINE.search(line)
        if m:
            self.found = int(m.group("n"))
            return
        m = _TEST_EXIT_LINE.search(line)
        if m:
            self.exit_code = int(m.group("c"))

    @classmethod
    def parse(cls, text: str) -> "AutomationLog":
        a = cls()
        for line in text.splitlines():
            a.feed(line)
        return a

    @classmethod
    def merge(cls, logs: list["AutomationLog"]) -> "AutomationLog":
        """Every source belongs to the current run; a test that failed anywhere counts as failed."""
        out = cls()
        for a in logs:
            for path, state in a.results.items():
                if path not in out.results or state.lower() not in SUCCESS_STATES:
                    out.add(state, path)
            if a.found is not None:
                out.found = a.found if out.found is None else max(out.found, a.found)
            if a.exit_code is not None and out.exit_code in (None, 0):
                out.exit_code = a.exit_code
        return out


def filter_matches(path: str, flt: str) -> bool:
    """Does a test's full path fall under the automation filter? Whole dotted segments, case-insensitive
    ("MyGame" matches MyGame.Car.Jump and Tests.MyGame.X, never MyGameOther.X); "A+B" means A or B."""
    segs = [x for x in path.lower().split(".") if x]
    for alt in (a.strip().lower() for a in flt.split("+")):
        want = [x for x in alt.split(".") if x]
        if want and any(segs[i:i + len(want)] == want for i in range(len(segs) - len(want) + 1)):
            return True
    return False


def automation_command(tool: list[str], project: str, flt: str, report: str) -> str | list[str]:
    """The headless automation run. -TestExit ends the editor when the test queue is empty; -stdout with
    -FullStdOutLogOutput puts every result line on stdout; the report goes to a directory created for this run
    (a result found there cannot be stale); the project log is written as usual (the project may rely on it)
    and read only as far as this run wrote it. On Windows Unreal reads -Key="value with spaces" from the raw
    command line (a whole quoted "-Key=value" token is not recognised), so the line is composed here from
    validated parts only (_quotable); elsewhere it is an argv list."""
    opts = [("ExecCmds", f"Automation RunTests {flt};Quit"), ("TestExit", "Automation Test Queue Empty"),
            ("ReportExportPath", report)]
    if sys.platform == "win32":
        return " ".join([subprocess.list2cmdline(tool), f'"{project}"', *(f'-{k}="{v}"' for k, v in opts),
                         *AUTOMATION_FLAGS])
    return [*tool, project, *(f"-{k}={v}" for k, v in opts), *AUTOMATION_FLAGS]
