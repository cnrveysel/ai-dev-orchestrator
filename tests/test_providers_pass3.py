"""V4 pass 3 capability providers: the generic runtime provider, the visual capture / analysis architecture,
the optional browser provider and the Unreal provider (with its build-log reducer).

Deterministic: local fixture services, a test-only fake capture backend and browser driver, and a MOCKED
Unreal engine tree whose tools are tiny Python fakes - no real Unreal installation, browser or network is
needed or touched. No external repository is used.
"""
import asyncio
import json
import os
import re
import socket
import struct
import sys
import time
import zlib
from pathlib import Path

import pytest

from app.evidence import procutil
from app.evidence.artifacts import ArtifactStore
from app.evidence.diagnostics import reduce_build_log
from app.evidence.executor import EvidenceExecutor, EvidenceLimits
from app.evidence.lifecycle import ResourceManager
from app.evidence.model import (
    FAILED, INCONCLUSIVE, INFERRED, PASSED, SATISFIED, UNAVAILABLE, CapabilitySnapshot, EvidenceState,
)
from app.evidence.planning import capability_line, validate_plan
from app.evidence.providers import unreal as ue
from app.evidence.providers.base import DiscoveryContext
from app.evidence.providers.browser import BrowserDriver, BrowserProvider, BrowserSession, PlaywrightDriver
from app.evidence.providers.command import CommandProvider
from app.evidence.providers.files import RepositoryFileProvider
from app.evidence.providers.runtime import RuntimeProvider
from app.evidence.providers.visual import CapturedImage, CaptureBackend, ImageAnalysis, ImageAnalyzer, VisualProvider
from app.evidence.registry import CapabilityRegistry, default_registry
from tests.test_evidence import make_phase, req
from tests.test_pipeline_v2 import git, make_settings

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def png(w: int, h: int) -> bytes:
    raw = b"".join(b"\x00" + b"\x10\x80\xf0" * w for _ in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def settings(**providers):
    o = make_settings().orchestration
    o.evidence_provider_settings = dict(providers)
    return o


async def discover_with(reg, repo, ph, o=None):
    return await reg.discover(DiscoveryContext(str(repo), ph, [ph], {}, o or settings()))


def executor(reg, repo, tmp_path, **limits):
    base = dict(wall_seconds=60, step_timeout=30, max_steps=12, ready_timeout=20)
    base.update(limits)
    return EvidenceExecutor(reg, EvidenceLimits(**base), ArtifactStore(str(tmp_path / "artifacts"), 1, "P1", 1,
                                                                       max_artifacts=24, max_bytes=1_000_000),
                            repo_path=str(repo), persist=lambda: None, log=lambda s: None)


def by_name(snap, name):
    return next(c for c in snap.capabilities if c.name == name)


def plan_of(ph, snap, reg, entries):
    return validate_plan({"summary": "s", "requirements": entries}, ph, snap, reg, 12)


# ============================================================ H. GENERIC INTERACTIVE RUNTIME PROVIDER

SMOKE = "import sys\nprint('SMOKE OK')\nsys.exit(0)\n"
DIE = "import time\nprint('UP', flush=True)\ntime.sleep(0.8)\nraise SystemExit(4)\n"
WAIT = "import time\ntime.sleep(1.5)\nprint('waited')\n"


@pytest.fixture
def rrepo(tmp_path):
    r = tmp_path / "app"
    (r / "web").mkdir(parents=True)
    (r / ".aistudio").mkdir()
    (r / "web" / "index.html").write_text("<html><body>hello runtime</body></html>")
    (r / "smoke.py").write_text(SMOKE)
    port = free_port()
    (r / ".aistudio" / "evidence.json").write_text(json.dumps({"runtimes": [{
        "name": "web", "command": f'"{PY}" -m http.server {port} --bind 127.0.0.1', "cwd": "web",
        "readiness": {"kind": "http_status", "url": f"http://127.0.0.1:{port}/index.html", "expected_status": 200},
        "ready_timeout_seconds": 15,
        "endpoints": [{"name": "index", "url": f"http://127.0.0.1:{port}/index.html"},
                      {"name": "remote", "url": "http://203.0.113.9/"}],
        "actions": [{"name": "smoke", "command": f'"{PY}" ../smoke.py', "kind": "runtime_behavior"}],
        "logs": [{"name": "server log", "path": "web/server.log"}]},
        {"name": "ghost", "command": "definitely-not-installed-xyz --serve"}]}))
    return r


def test_runtime_discovery_declares_observing_capabilities_honestly(rrepo):
    reg = CapabilityRegistry([RuntimeProvider()])
    snap = asyncio.run(discover_with(reg, rrepo, make_phase([req("E1")])))
    by = {c.key: c for c in snap.capabilities}
    launch = by["runtime:web"]
    assert launch.kind == "runtime_launch" and launch.confidence == INFERRED
    assert launch.constraints["lifecycle"] == "service" and launch.constraints["readiness"]["kind"] == "http_status"
    for suffix, kind in (("ready", "runtime_ready"), ("alive", "process_control"), ("log", "runtime_log"),
                         ("endpoint:index", "local_endpoint"), ("action:smoke", "runtime_behavior")):
        c = by[f"runtime:web:{suffix}"]
        assert c.kind == kind and c.confidence == INFERRED and c.constraints["requires_resource"] == "runtime:web"
    assert by["runtime:web:endpoint:remote"].confidence == UNAVAILABLE
    assert "limited to this machine" in by["runtime:web:endpoint:remote"].reason
    ghost = [c for c in snap.capabilities if c.key.startswith("runtime:ghost")]
    assert ghost and all(c.confidence == UNAVAILABLE for c in ghost), "a runtime that cannot start offers nothing"
    assert "observes a running resource: list" in capability_line(by["runtime:web:alive"], snap)


def test_runtime_is_launched_probed_observed_and_stopped(rrepo, tmp_path):
    reg = CapabilityRegistry([RuntimeProvider()])
    ph = make_phase([req("E1"), req("E2")])
    snap = asyncio.run(discover_with(reg, rrepo, ph))
    k = {c.key: c.id for c in snap.capabilities}
    plan = plan_of(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct", "resources": [k["runtime:web"]],
         "steps": [{"capability_id": k["runtime:web:endpoint:index"], "expect_output": "hello runtime"},
                   {"capability_id": k["runtime:web:action:smoke"], "expect_output": "SMOKE OK"},
                   {"capability_id": k["runtime:web:log"], "fail_output": "Traceback"},
                   {"capability_id": k["runtime:web:alive"]},
                   {"capability_id": k["runtime:web:ready"]}]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": k["runtime:web:alive"]}]}])  # no runtime started for it: rejected
    assert "which is not listed in `resources`" in " ".join(plan.get("E2").notes)
    assert plan.get("E2").strategy == "unavailable"
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor(reg, rrepo, tmp_path).run(state, plan, snap))
    by = {r.requirement_id: r for r in bundle.results}
    assert by["E1"].status == SATISFIED, [s.get("summary") for s in state.steps.values()]
    assert [state.steps[f"E1.s{i}"]["status"] for i in range(1, 6)] == [PASSED] * 5
    assert "HTTP 200 and contains 'hello runtime'" in state.steps["E1.s1"]["summary"]
    assert state.resources[0]["state"] == "STOPPED" and not procutil.is_alive(state.resources[0]["pid"]), \
        "the runtime is always stopped - no leaked process"
    assert any("while E1 was observed" in a.description for a in bundle.artifacts)


def test_a_runtime_that_dies_is_caught_and_stale_runtimes_are_reaped(tmp_path):
    r = tmp_path / "app"
    (r / ".aistudio").mkdir(parents=True)
    (r / "die.py").write_text(DIE)
    (r / "wait.py").write_text(WAIT)
    (r / ".aistudio" / "evidence.json").write_text(json.dumps({"runtimes": [{
        "name": "fragile", "command": f'"{PY}" die.py', "readiness": {"kind": "stdout_contains", "text": "UP"},
        "actions": [{"name": "wait", "command": f'"{PY}" wait.py'}]}]}))
    reg = CapabilityRegistry([RuntimeProvider()])
    ph = make_phase([req("E1")])
    snap = asyncio.run(discover_with(reg, r, ph))
    k = {c.key: c.id for c in snap.capabilities}
    plan = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                    "resources": [k["runtime:fragile"]],
                                    "steps": [{"capability_id": k["runtime:fragile:action:wait"]},
                                              {"capability_id": k["runtime:fragile:alive"]}]}])
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor(reg, r, tmp_path).run(state, plan, snap))
    assert bundle.results[0].status == FAILED and bundle.results[0].cause == "product"
    assert state.steps["E1.s1"]["reason"] == "the runtime crashed while the action ran"
    assert "exited with code 4" in state.steps["E1.s2"]["summary"]
    # a runtime left running by a crashed collection is reaped only when provably ours
    import subprocess
    proc = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(0.3)
        st = EvidenceState("P1", resources=[{"id": "res-1", "capability_id": k["runtime:fragile"], "pid": proc.pid,
                                             "start_time": procutil.start_time(proc.pid), "state": "READY"}])
        notes = ResourceManager(st, lambda: None, lambda s: None).reap_stale()
        assert st.resources[0]["state"] == "REAPED" and "killed" in notes[0]
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()


# ============================================================ I. VISUAL CAPTURE / ANALYSIS

class FakeCapture(CaptureBackend):
    """TEST-ONLY capture backend: a deterministic image of a running handle (never used by the app)."""
    name = "fake-capture"

    def __init__(self):
        self.calls = 0

    def available(self):
        return True, "test capture backend"

    async def capture(self, handle, *, timeout):
        self.calls += 1
        return CapturedImage(png(8, 4), 8, 4, description=f"test image of pid {getattr(handle, 'pid', '?')}")


class FakeAnalyzer(ImageAnalyzer):
    name = "fake-analyzer"

    def __init__(self, verdict=True):
        self.verdict, self.seen = verdict, []

    def available(self):
        return True, "test analyzer"

    async def analyze(self, image_path, *, question, timeout):
        self.seen.append((image_path, question))
        data = Path(image_path).read_bytes()
        assert data.startswith(b"\x89PNG"), "analysis reads the captured artifact"
        return ImageAnalysis(self.verdict, "the hierarchy is clear" if self.verdict else "the text is unreadable")


@pytest.fixture
def vrepo(tmp_path):
    r = tmp_path / "vis"
    (r / ".aistudio").mkdir(parents=True)
    (r / "app.py").write_text("import time\nprint('UP', flush=True)\ntime.sleep(30)\n")
    (r / "build.py").write_text("print('build ok')\n")
    (r / ".aistudio" / "evidence.json").write_text(json.dumps({
        "commands": [{"name": "build", "command": f'"{PY}" build.py', "kind": "build"}],
        "runtimes": [{"name": "app", "command": f'"{PY}" app.py',
                      "readiness": {"kind": "stdout_contains", "text": "UP"}}]}))
    return r


def test_without_a_capture_backend_there_is_no_visual_evidence(vrepo, tmp_path):
    reg = CapabilityRegistry([CommandProvider(), RepositoryFileProvider(), RuntimeProvider(), VisualProvider()])
    ph = make_phase([req("E1", kind="visual_evidence", goal="the layout has a clear hierarchy")])
    snap = asyncio.run(discover_with(reg, vrepo, ph))
    cap, ana = (by_name(snap, "visual capture of a running resource"), by_name(snap, "image analysis"))
    assert cap.confidence == UNAVAILABLE and "no screenshot/capture backend" in cap.reason
    assert ana.confidence == UNAVAILABLE and "never visual evidence" in ana.reason
    plan = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                    "resources": [by_name(snap, "runtime: app").id],
                                    "steps": [{"capability_id": cap.id}]}])
    assert plan.get("E1").strategy == "unavailable", "an unavailable capture cannot be planned"
    bundle = asyncio.run(executor(reg, vrepo, tmp_path).run(EvidenceState("P1", attempt=1), plan, snap))
    assert bundle.results[0].status == UNAVAILABLE and not bundle.artifacts, "nothing is ever fabricated"
    build = by_name(snap, "build")
    plan2 = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                     "steps": [{"capability_id": build.id}]}])
    assert plan2.get("E1").coverage == "indirect", "a green build is not visual evidence"


def test_capture_and_analysis_are_distinct_capabilities(vrepo, tmp_path):
    fake_cap, analyzer = FakeCapture(), FakeAnalyzer()
    reg = CapabilityRegistry([CommandProvider(), RepositoryFileProvider(), RuntimeProvider(),
                              VisualProvider(fake_cap, analyzer)])
    ph = make_phase([req("E1", kind="visual_evidence", goal="clear hierarchy"),
                     req("E2", kind="visual_evidence", goal="clear hierarchy")])
    snap = asyncio.run(discover_with(reg, vrepo, ph))
    app, cap, ana = (by_name(snap, "runtime: app").id, by_name(snap, "visual capture of a running resource").id,
                     by_name(snap, "image analysis").id)
    plan = plan_of(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct", "resources": [app],
         "steps": [{"capability_id": cap}]},  # capture only
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct", "resources": [app],
         "steps": [{"capability_id": cap}, {"capability_id": ana, "args": ["1"],
                                            "expect_output": "Is the hierarchy clear?"}]}])
    assert plan.get("E1").coverage == "indirect" and "visual_capture does not observe" in plan.get("E1").notes[0]
    assert plan.get("E2").steps[1].args == ["E2.s1"], "the analysis consumes the capture step of its requirement"
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor(reg, vrepo, tmp_path).run(state, plan, snap))
    by = {r.requirement_id: r for r in bundle.results}
    assert by["E1"].status == INCONCLUSIVE and "supporting evidence only" in by["E1"].summary, \
        "a screenshot exists != the visual result is good"
    assert by["E2"].status == SATISFIED and "the hierarchy is clear" in by["E2"].summary
    images = [a for a in bundle.artifacts if a.media_type == "image/png"]
    assert len(images) == 2 and all(Path(a.path_or_ref).read_bytes().startswith(b"\x89PNG") for a in images)
    m = images[0].metadata
    assert (m["width"], m["height"], m["requirement_id"], m["phase_id"], m["source_capability"]) == (8, 4, "E1", "P1", cap)
    assert m["captured_at"] > 0 and m["resource"] == "runtime:app"
    restored = CapabilitySnapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
    assert restored.get(cap).kind == "visual_capture"
    again = EvidenceState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert again.artifacts[0]["media_type"] == "image/png" and again.artifacts[0]["metadata"]["width"] == 8, \
        "image metadata persists"
    assert analyzer.seen and analyzer.seen[0][1] == "Is the hierarchy clear?"
    bad = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                   "resources": [app], "steps": [{"capability_id": ana, "args": ["1"]}]}])
    assert "must be the number of an earlier step" in " ".join(bad.get("E1").notes)


TK_WINDOW = ("import tkinter as tk\n"
             "r = tk.Tk()\n"
             "r.title('AIStudio capture fixture')\n"
             "r.geometry('240x140+-4000+-4000')\n"  # off-screen: never shown to the user
             "tk.Label(r, text='HELLO CAPTURE', bg='#20c040', fg='white').pack(fill='both', expand=True)\n"
             "r.update()\n"
             "print('UP', flush=True)\n"
             "r.after(30000, r.destroy)\n"
             "r.mainloop()\n")


@pytest.mark.skipif(sys.platform != "win32", reason="the window capture backend is Windows-only")
def test_real_window_capture_of_a_started_resource_is_an_honest_image_artifact(tmp_path):
    pytest.importorskip("tkinter")
    r = tmp_path / "ui"
    (r / ".aistudio").mkdir(parents=True)
    (r / "window.py").write_text(TK_WINDOW)
    (r / ".aistudio" / "evidence.json").write_text(json.dumps({"runtimes": [{
        "name": "window", "command": f'"{PY}" window.py', "readiness": {"kind": "stdout_contains", "text": "UP"}}]}))
    reg = default_registry()
    ph = make_phase([req("E1", kind="visual_evidence", goal="the start screen is readable")])
    snap = asyncio.run(discover_with(reg, r, ph))
    cap = by_name(snap, "visual capture of a running resource")
    assert cap.confidence == INFERRED and cap.details["backend"] == "windows-window"
    assert by_name(snap, "image analysis").confidence == UNAVAILABLE, "capture does not imply analysis"
    plan = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                    "resources": [by_name(snap, "runtime: window").id],
                                    "steps": [{"capability_id": cap.id}]}])
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor(reg, r, tmp_path).run(state, plan, snap))
    result = bundle.results[0]
    assert result.status == INCONCLUSIVE and "captured only, not analysed" in result.summary, \
        "a real screenshot is still not a judgment of how it looks"
    [img] = [a for a in bundle.artifacts if a.media_type == "image/png"]
    data = Path(img.path_or_ref).read_bytes()
    w, h = struct.unpack(">II", data[16:24])
    assert data.startswith(b"\x89PNG") and (w, h) == (img.metadata["width"], img.metadata["height"]) and w >= 200
    assert img.metadata["window_title"] == "AIStudio capture fixture", "only the started resource's window"
    assert not procutil.is_alive(state.resources[0]["pid"])


def test_a_failed_analysis_is_a_finding_not_a_pass(vrepo, tmp_path):
    reg = CapabilityRegistry([RuntimeProvider(), VisualProvider(FakeCapture(), FakeAnalyzer(verdict=False))])
    ph = make_phase([req("E1", kind="visual_evidence")])
    snap = asyncio.run(discover_with(reg, vrepo, ph))
    plan = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                    "resources": [by_name(snap, "runtime: app").id],
                                    "steps": [{"capability_id": by_name(snap, "visual capture of a running resource").id},
                                              {"capability_id": by_name(snap, "image analysis").id, "args": ["1"]}]}])
    bundle = asyncio.run(executor(reg, vrepo, tmp_path).run(EvidenceState("P1", attempt=1), plan, snap))
    assert bundle.results[0].status == FAILED and "the text is unreadable" in bundle.results[0].summary


# ============================================================ J. BROWSER / UI PROVIDER (optional)

class FakeSession(BrowserSession):
    def __init__(self, driver):
        self.driver, self.html, self.clicked = driver, "", []

    async def goto(self, url, timeout):
        self.driver.visited.append(url)
        if url.startswith("file:"):
            self.html = Path(url.replace("file:///", "")).read_text()
            return None
        return 200

    async def click(self, selector, timeout):
        if f"id='{selector.lstrip('#')}'" not in self.html:
            raise RuntimeError(f"no element {selector}")
        self.clicked.append(selector)
        if selector == "#start":
            self.html += "<p id='score'>score 1</p>"

    async def press(self, key):
        self.clicked.append(f"key:{key}")

    async def type(self, selector, text, timeout):
        if "password" in selector:
            raise PermissionError("typing into password fields is not allowed")
        self.html += text

    async def text(self, selector=""):
        return re.sub(r"<[^>]+>", " ", self.html)

    async def screenshot(self):
        return png(6, 3), 6, 3

    def console(self):
        return ["log: page loaded", "error: Uncaught TypeError: x is undefined"] if "buggy" in self.html else \
            ["log: page loaded"]

    async def close(self):
        self.driver.closed += 1


class FakeDriver(BrowserDriver):
    name = "fake"

    def __init__(self):
        self.visited, self.opened, self.closed = [], 0, 0

    def available(self):
        return True, "test driver"

    async def open(self, timeout):
        self.opened += 1
        return FakeSession(self)


@pytest.fixture
def brepo(tmp_path):
    r = tmp_path / "site"
    (r / ".aistudio").mkdir(parents=True)
    (r / "index.html").write_text("<html><body><button id='start'>Start</button><input id='name'>"
                                  "<input id='password' type='password'></body></html>")
    (r / "buggy.html").write_text("<html><body>buggy page</body></html>")
    (r / ".aistudio" / "evidence.json").write_text(json.dumps({"browser": {"pages": [
        {"name": "home", "path": "index.html"}, {"name": "buggy", "path": "buggy.html"},
        {"name": "remote", "url": "https://example.com/"}]}}))
    return r


def test_the_browser_provider_is_off_by_default_and_honest_without_a_driver(brepo):
    ph = make_phase([req("E1", kind="interaction")])
    assert asyncio.run(discover_with(CapabilityRegistry([BrowserProvider(FakeDriver())]), brepo, ph)).capabilities == []
    snap = asyncio.run(discover_with(CapabilityRegistry([BrowserProvider(PlaywrightDriver())]), brepo, ph,
                                     settings(**{"browser.enabled": "true"})))
    ok, why = PlaywrightDriver().available()
    if not ok:  # this environment has no Playwright: every capability says so (the remote page is refused anyway)
        assert snap.capabilities and all(c.confidence == UNAVAILABLE for c in snap.capabilities)
        assert all(why in c.reason for c in snap.capabilities if not c.key.startswith("browser:remote"))


def test_browser_capabilities_run_local_pages_with_bounded_actions_and_always_close(brepo, tmp_path):
    driver = FakeDriver()
    reg = CapabilityRegistry([BrowserProvider(driver)])
    ph = make_phase([req("E1", kind="interaction"), req("E2", kind="interaction"), req("E3", kind="console"),
                     req("E4", kind="visual_evidence"), req("E5", kind="interaction")])
    snap = asyncio.run(discover_with(reg, brepo, ph, settings(**{"browser.enabled": "true"})))
    k = {c.key: c for c in snap.capabilities}
    assert k["browser:remote:probe"].confidence == UNAVAILABLE and "localhost" in k["browser:remote:probe"].reason
    assert k["browser:home:ui"].confidence == INFERRED
    ui, home_console = k["browser:home:ui"].id, k["browser:buggy:console"].id
    plan = plan_of(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": ui, "args": ["click:#start", "expect:#score=score 1"]}]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": ui, "args": ["expect:game over"]}]},
        {"requirement_id": "E3", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": home_console, "fail_output": "Uncaught"}]},
        {"requirement_id": "E4", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": k["browser:home:screenshot"].id}]},
        {"requirement_id": "E5", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": ui, "args": ["type:#password=hunter2"]}]}])
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor(reg, brepo, tmp_path).run(state, plan, snap))
    st = {r.requirement_id: r.status for r in bundle.results}
    assert st == {"E1": SATISFIED, "E2": FAILED, "E3": FAILED, "E4": INCONCLUSIVE, "E5": UNAVAILABLE}
    assert "not allowed" in bundle.results[4].summary, "no credential entry"
    assert any(a.media_type == "image/png" for a in bundle.artifacts), "the screenshot is an artifact (not a verdict)"
    assert driver.opened == driver.closed == 5, "every page/browser is closed"
    assert all(u.startswith("file:") for u in driver.visited), "only the declared local pages were visited"
    bad = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g",
                                   "steps": [{"capability_id": ui, "args": ["eval:document.cookie"]}]}])
    assert "invalid action" in " ".join(bad.get("E1").notes)


# ============================================================ K. UNREAL PROVIDER (mocked engine)

FAKE_UBT = r'''
import os, sys
mode = os.environ.get("FAKE_UBT_MODE", "ok")
print("Using bundled DotNet SDK")
print("Building %s and ShaderCompileWorker..." % sys.argv[1])
print("Parsing headers for %s" % sys.argv[1])
if mode == "uht":
    print(r"D:\Proj\Source\MyGame\Public\Car.h(12): Error: Unrecognized type 'FWheel' - type must be a UCLASS, USTRUCT or UENUM")
    print(r"D:\Proj\Source\MyGame\Private\Car.cpp(40): error C2065: 'Wheels': undeclared identifier")
    print("UnrealHeaderTool failed for target 'MyGameEditor'")
    print("Result: Failed (OtherCompilationError)")
    sys.exit(6)
if mode == "live":
    print("Unable to build while Live Coding is active. Exit the editor and game, or press Ctrl+Alt+F11")
    sys.exit(6)
print("[1/3] Compile Module.MyGame.cpp")
print("Result: Succeeded")
sys.exit(0)
'''
FAKE_EDITOR = r'''
import os, sys, time
mode = os.environ.get("FAKE_EDITOR_MODE", "ok")
print("LogInit: Display: Running engine for game: MyGame", flush=True)
print("LogLoad: Took 0.42 seconds to LoadMap(/Game/Maps/Main)", flush=True)
if mode == "crash":
    time.sleep(0.3)
    print("LogWindows: Error: === Critical error: ===", flush=True)
    print("Fatal error!", flush=True)
time.sleep(30)
'''
FAKE_EDITOR_CMD = r'''
import os, sys, json
# A mocked UnrealEditor-Cmd automation run. FAKE_AUTOMATION_MODE picks what it reports and WHERE:
#   ok / fail      stdout + report          stdout      stdout only (no report)
#   project_log / project_log_fail   only APPENDED to Saved/Logs/<name>.log
#   project_log_rotate               the old log renamed to a backup (as Unreal does), results only in a NEW log
#   silent         nothing, exit 0          none        "Found 0 automation tests"      unrelated   another game's test
mode = os.environ.get("FAKE_AUTOMATION_MODE", "ok")
if os.environ.get("FAKE_ARGV_FILE"):
    json.dump(sys.argv[1:], open(os.environ["FAKE_ARGV_FILE"], "w"))
def opt(k):
    return next((a.split("=", 1)[1] for a in sys.argv if a.lower().startswith("-" + k.lower() + "=")), "")
cmds = opt("ExecCmds")
flt = cmds.split("RunTests", 1)[1].split(";")[0].strip() if "RunTests" in cmds else ""
print("LogAutomationCommandLine: Display: -ExecCmds=" + cmds)
if mode == "silent":
    sys.exit(0)
if mode == "none":
    print("LogAutomationCommandLine: Display: Found 0 automation tests based on '%s'" % flt)
    sys.exit(0)
tests = {"unrelated": [("Success", "OtherGame.Menu.Opens")],
         "fail": [("Success", "MyGame.Car.Wheels"), ("Fail", "MyGame.Car.Jump")],
         "project_log_fail": [("Fail", "MyGame.Car.Wheels")]}.get(
    mode, [("Success", "MyGame.Car.Wheels"), ("Success", "MyGame.Car.Jump")])
lines = ["LogAutomationCommandLine: Display: Found %d automation tests based on '%s'" % (len(tests), flt)]
lines += ["LogAutomationCommandLine: Display: \t%s" % p for _s, p in tests]
lines += ["LogAutomationController: Display: Test Completed. Result={%s} Name={%s} Path={%s}"
          % (st, p.rsplit(".", 1)[1], p) for st, p in tests]
lines += ["LogAutomationController: Display: **** TEST COMPLETE. EXIT CODE: 0 ****"]
if mode.startswith("project_log"):
    logs = os.path.join(os.path.dirname(sys.argv[1]), "Saved", "Logs")
    os.makedirs(logs, exist_ok=True)
    log = os.path.join(logs, os.path.splitext(os.path.basename(sys.argv[1]))[0] + ".log")
    if mode == "project_log_rotate" and os.path.exists(log):
        os.replace(log, log[:-4] + "-backup-2026.01.01-00.00.00.log")
    open(log, "a").write("Log file open, new run\n" + "\n".join(lines) + "\n")
else:
    print("\n".join(lines))
    if mode in ("ok", "fail", "unrelated"):
        json.dump({"tests": [{"state": st, "fullTestPath": p} for st, p in tests]},
                  open(os.path.join(opt("ReportExportPath"), "index.json"), "w"))
'''


@pytest.fixture
def uproject(tmp_path):
    r = tmp_path / "MyGame"
    (r / "Source" / "MyGame").mkdir(parents=True)
    (r / "MyGame.uproject").write_text(json.dumps({"FileVersion": 3, "EngineAssociation": "5.4", "Modules": [
        {"Name": "MyGame", "Type": "Runtime", "LoadingPhase": "Default"}]}))
    (r / "Source" / "MyGame.Target.cs").write_text("// game target\n")
    (r / "Source" / "MyGameEditor.Target.cs").write_text("// editor target\n")
    (r / "Source" / "MyGame" / "MyGame.Build.cs").write_text("// module\n")
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "i")
    return r


@pytest.fixture
def engine(tmp_path):
    root = tmp_path / "Engines" / "UE_5.4"
    (root / "Engine" / "Build").mkdir(parents=True)
    (root / "Engine" / "Build" / "Build.version").write_text(json.dumps({"MajorVersion": 5, "MinorVersion": 4,
                                                                         "PatchVersion": 2}))
    ubt = root / "Engine" / "Binaries" / "DotNET" / "UnrealBuildTool" / "UnrealBuildTool.exe"
    ubt.parent.mkdir(parents=True)
    ubt.write_bytes(b"MZ fake")
    win = root / "Engine" / "Binaries" / "Win64"
    win.mkdir(parents=True)
    (win / "UnrealEditor.exe").write_bytes(b"MZ fake")
    (win / "UnrealEditor-Cmd.exe").write_bytes(b"MZ fake")
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    (fakes / "UnrealBuildTool.py").write_text(FAKE_UBT)
    (fakes / "UnrealEditor.py").write_text(FAKE_EDITOR)
    (fakes / "UnrealEditor-Cmd.py").write_text(FAKE_EDITOR_CMD)
    return root, fakes


def unreal_provider(fakes=None, locator=None):
    prefix = (lambda path: [PY, str(fakes / (Path(path).stem + ".py"))]) if fakes else None
    return ue.UnrealProvider(locator=locator, tool_prefix=prefix)


def test_a_non_unreal_project_gets_no_unreal_capabilities(tmp_path):
    (tmp_path / "plain").mkdir()
    snap = asyncio.run(discover_with(CapabilityRegistry([ue.UnrealProvider()]), tmp_path / "plain",
                                     make_phase([req("E1")])))
    assert snap.capabilities == [] and snap.providers["unreal"] == "ok (0)"


def test_an_unreal_project_without_an_engine_is_reported_unavailable(uproject):
    reg = CapabilityRegistry([unreal_provider(locator=lambda p, repo, root: (None, "no engine for '5.4' (test)"))])
    snap = asyncio.run(discover_with(reg, uproject, make_phase([req("E1", kind="build")])))
    by = {c.key.split(":", 2)[-1]: c for c in snap.capabilities}
    assert by["project"].confidence == INFERRED and by["project"].kind == "static_analysis"
    for k in ("build", "automation", "game", "editor", "log", "crash"):
        assert by[k].confidence == UNAVAILABLE and "Unreal Engine unavailable" in by[k].reason, k
    assert by["uht"].confidence == UNAVAILABLE and "runs as part of the build" in by["uht"].reason
    assert by["performance"].confidence == UNAVAILABLE and "not implemented" in by["performance"].reason
    missing = asyncio.run(discover_with(CapabilityRegistry([ue.UnrealProvider()]), uproject,
                                        make_phase([req("E1")]),
                                        settings(**{"unreal.engine_root": str(uproject / "no-engine-here")})))
    build = next(c for c in missing.capabilities if c.key.endswith(":build"))
    assert build.confidence == UNAVAILABLE and "is not an Unreal Engine directory" in build.reason


def test_a_mocked_engine_install_is_discovered_with_its_tools(uproject, engine):
    root, _fakes = engine
    reg = CapabilityRegistry([ue.UnrealProvider()])
    snap = asyncio.run(discover_with(reg, uproject, make_phase([req("E1", kind="build")]),
                                     settings(**{"unreal.engine_root": str(root), "unreal.test_filter": "MyGame"})))
    by = {c.key.split(":", 2)[-1]: c for c in snap.capabilities}
    for k in ("project", "build", "automation", "game", "editor", "log", "crash"):
        assert by[k].confidence == INFERRED, (k, by[k].reason)
    assert by["build"].kind == "build" and by["build"].details["target"] == "MyGameEditor"
    assert by["build"].details["engine_version"] == "5.4.2" and "provider setting" in by["build"].provenance["engine"]
    assert by["automation"].details["default_filter"] == "MyGame"
    assert by["game"].constraints["lifecycle"] == "service" and by["log"].constraints["requires_resource"] == "*"
    assert {c.kind for c in snap.capabilities}.isdisjoint({"blueprint_authoring", "blueprint_graph"})


def test_the_build_command_is_an_argv_list_built_from_validated_paths(uproject, engine):
    root, _ = engine
    e = ue.engine_from_root(str(root), "test")
    p = ue.read_project(str(uproject))
    argv = ue.build_argv(e, p, "MyGameEditor", "Win64", "Development")
    assert isinstance(argv, list) and argv[0] == e.ubt, "no shell: an argv list"
    assert argv[1:] == ["MyGameEditor", "Win64", "Development", f"-Project={p.path}", "-WaitMutex",
                        "-NoHotReloadFromIDE"]
    assert not ue._quotable('C:\\evil" & del *') and ue._quotable("C:\\Program Files\\Epic Games\\UE_5.4")


def test_unreal_build_failures_are_reduced_to_root_diagnostics(uproject, engine, tmp_path, monkeypatch):
    root, fakes = engine
    reg = CapabilityRegistry([unreal_provider(fakes)])
    ph = make_phase([req("E1", kind="build")])
    snap = asyncio.run(discover_with(reg, uproject, ph, settings(**{"unreal.engine_root": str(root)})))
    build = next(c for c in snap.capabilities if c.key.endswith(":build"))
    plan = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                    "steps": [{"capability_id": build.id}]}])
    monkeypatch.setenv("FAKE_UBT_MODE", "uht")
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor(reg, uproject, tmp_path).run(state, plan, snap))
    r = bundle.results[0]
    assert r.status == FAILED and "failed in the header tool (UHT)" in r.summary
    assert r.diagnostics[0].startswith("build failed (MyGameEditor Win64 Development)")
    assert "the header tool (UHT) failed" in r.diagnostics[0]
    assert r.diagnostics[1].startswith("UHT error MyGame/Public/Car.h(12): Unrecognized type 'FWheel'"), \
        "the root cause (header tool) comes first; paths outside the project keep their last components"
    assert any("C2065" in x for x in r.diagnostics)
    assert Path(bundle.artifacts[0].path_or_ref).read_text().count("Parsing headers") == 1, "the full log is kept"
    monkeypatch.setenv("FAKE_UBT_MODE", "live")
    b2 = asyncio.run(executor(reg, uproject, tmp_path).run(EvidenceState("P1", attempt=1), plan, snap))
    assert b2.results[0].status == UNAVAILABLE and b2.results[0].cause == "strategy", \
        "a blocked build environment is not a code failure"
    monkeypatch.setenv("FAKE_UBT_MODE", "ok")
    b3 = asyncio.run(executor(reg, uproject, tmp_path).run(EvidenceState("P1", attempt=1), plan, snap))
    assert b3.results[0].status == SATISFIED and "succeeded" in b3.results[0].summary


def test_unreal_automation_tests_and_runtime_crash_detection(uproject, engine, tmp_path, monkeypatch):
    root, fakes = engine
    reg = CapabilityRegistry([unreal_provider(fakes)])
    ph = make_phase([req("E1", kind="automated_test"), req("E2", kind="runtime_behavior")])
    snap = asyncio.run(discover_with(reg, uproject, ph, settings(**{"unreal.engine_root": str(root)})))
    k = {c.key.split(":", 2)[-1]: c.id for c in snap.capabilities}
    plan = plan_of(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": k["automation"], "args": ["MyGame.Car"]}]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct", "resources": [k["game"]],
         "steps": [{"capability_id": k["crash"], "args": ["1"]},
                   {"capability_id": k["log"], "expect_output": "LoadMap"}]}])
    bad = plan_of(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g",
                                   "steps": [{"capability_id": k["automation"], "args": ["x; rm -rf"]}]}])
    assert bad.get("E1").strategy == "unavailable", "the test filter is validated"
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor(reg, uproject, tmp_path).run(state, plan, snap))
    by = {r.requirement_id: r for r in bundle.results}
    assert by["E1"].status == SATISFIED and "2 automation test(s) matching 'MyGame.Car' passed" in by["E1"].summary
    assert by["E2"].status == SATISFIED and "with no fatal error" in state.steps["E2.s1"]["summary"]
    assert state.resources[0]["state"] == "STOPPED" and not procutil.is_alive(state.resources[0]["pid"])
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "fail")
    monkeypatch.setenv("FAKE_EDITOR_MODE", "crash")
    state2 = EvidenceState("P1", attempt=1)
    b2 = asyncio.run(executor(reg, uproject, tmp_path).run(state2, plan, snap))
    by2 = {r.requirement_id: r for r in b2.results}
    assert by2["E1"].status == FAILED and "1 of 2 automation test(s) matching 'MyGame.Car' failed" in by2["E1"].summary
    assert "failed: MyGame.Car.Jump" in by2["E1"].diagnostics
    assert by2["E2"].status == FAILED and "crashed or stopped: fatal marker" in state2.steps["E2.s1"]["summary"]
    assert not procutil.is_alive(state2.resources[0]["pid"])
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "none")
    b3 = asyncio.run(executor(reg, uproject, tmp_path).run(EvidenceState("P1", attempt=1), plan_of(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": k["automation"], "args": ["Nothing.Here"]}]}]), snap))
    assert b3.results[0].status == FAILED and "no automation test matches 'Nothing.Here'" in b3.results[0].summary, \
        "the run itself reports that no test exists: evidence of a missing test, not an unusable capability"


def test_unreal_commandlets_are_declared_read_only_and_never_author_blueprints(uproject, engine):
    root, _ = engine
    (uproject / ".aistudio").mkdir()
    (uproject / ".aistudio" / "evidence.json").write_text(json.dumps({"unreal": {"commandlets": [
        {"name": "compile blueprints", "commandlet": "CompileAllBlueprints", "kind": "static_analysis"},
        {"name": "resave", "commandlet": "ResavePackages"},
        {"name": "bad args", "commandlet": "DataValidation", "args": ["-Paths=/Game; del *"]}]}}))
    snap = asyncio.run(discover_with(CapabilityRegistry([ue.UnrealProvider()]), uproject, make_phase([req("E1")]),
                                     settings(**{"unreal.engine_root": str(root)})))
    by = {c.name: c for c in snap.capabilities}
    assert by["Unreal commandlet: compile blueprints"].confidence == INFERRED
    assert by["Unreal commandlet: resave"].confidence == UNAVAILABLE and "never used for verification" in \
        by["Unreal commandlet: resave"].reason
    assert by["Unreal commandlet: bad args"].confidence == UNAVAILABLE
    src = (ROOT / "app" / "evidence" / "providers" / "unreal.py").read_text(encoding="utf-8")
    for forbidden in ("K2Node", "EdGraph", "BlueprintEditorLibrary", "AddFunctionGraph", "unreal.EditorAssetLibrary"):
        assert forbidden not in src, "no Blueprint graph authoring"


# ============================================================ L. BUILD-LOG REDUCER (fixture logs)

MSVC_LOG = """Building MyGameEditor...
[1/4] Compile Module.MyGame.cpp
D:\\Proj\\Source\\MyGame\\Private\\Car.cpp(40,12): error C2065: 'Wheels': undeclared identifier
D:\\Proj\\Source\\MyGame\\Private\\Car.cpp(40,12): error C2065: 'Wheels': undeclared identifier
D:\\Proj\\Source\\MyGame\\Private\\Car.cpp(41,1): warning C4996: 'Old': deprecated
Module.MyGame.cpp.obj : error LNK2019: unresolved external symbol "void __cdecl Spin(void)"
LINK : fatal error LNK1120: 1 unresolved externals
Result: Failed (OtherCompilationError)
"""
CLANG_LOG = """src/car.cpp:12:5: error: use of undeclared identifier 'wheels'
src/car.cpp:30:1: warning: unused variable 'x'
make: *** [car.o] Error 1
"""
UHT_LOG = """Parsing headers for MyGameEditor
D:\\Proj\\Source\\MyGame\\Public\\Car.h(12): Error: Unrecognized type 'FWheel'
D:\\Proj\\Source\\MyGame\\Private\\Car.cpp(3): error C1083: Cannot open include file: 'Car.generated.h'
ERROR: UnrealHeaderTool failed for target 'MyGameEditor'
"""


def test_the_build_log_reducer_extracts_compact_root_errors():
    s = reduce_build_log(MSVC_LOG, root="D:\\Proj", platform="Win64", configuration="Development")
    assert s.result == "failed" and s.target == "MyGameEditor" and s.error_count == 4 and s.warning_count == 1
    lines = s.lines()
    assert lines[0] == "build failed (MyGameEditor Win64 Development): 4 error(s), 3 distinct, 1 warning(s)"
    assert lines[1] == "compiler error C2065 Source/MyGame/Private/Car.cpp(40): 'Wheels': undeclared identifier (x2)"
    assert any(x.startswith("linker error LNK2019") for x in lines) and any("LNK1120" in x for x in lines)
    c = reduce_build_log(CLANG_LOG)
    assert c.errors[0].line_text() == "compiler error src/car.cpp(12): use of undeclared identifier 'wheels'"
    assert c.warning_count == 1
    u = reduce_build_log(UHT_LOG, root="D:\\Proj")
    assert u.uht_failed and u.errors[0].category == "uht" and u.errors[0].file == "Source/MyGame/Public/Car.h"
    assert [d.category for d in u.errors] == ["uht", "compiler", "build_tool"], "root cause first"
    ok = reduce_build_log("Building X...\nResult: Succeeded\n")
    assert ok.result == "succeeded" and not ok.errors
    many = reduce_build_log("\n".join(f"a.cpp({i}): error C2143: syntax error" for i in range(500)))
    assert many.error_count == 500 and len(many.lines(8)) == 10 and "more distinct error(s)" in many.lines(8)[-1]


# ============================================================ architecture: no technology in the orchestration

def test_the_scheduler_and_the_evidence_core_contain_no_technology_branches():
    core = ["app/orchestration/phase_scheduler.py", "app/orchestration/scheduler.py", "app/orchestration/checkpoint.py",
            "app/orchestration/consistency.py", "app/orchestration/impl_guard.py", "app/evidence/executor.py",
            "app/evidence/planning.py", "app/evidence/model.py", "app/evidence/freshness.py",
            "app/evidence/registry.py", "app/evidence/lifecycle.py"]
    tech = re.compile(r"\b(unreal|uproject|blueprint|browser|playwright|chrome|chromium|selenium|threejs|usb|"
                      r"sausage)\b", re.IGNORECASE)
    for rel in core:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.match(r"\s*(if|elif|while)\b", code) and tech.search(code):
                raise AssertionError(f"{rel}:{n}: technology-specific conditional: {line.strip()}")
    registry = (ROOT / "app/evidence/registry.py").read_text(encoding="utf-8")
    assert "UnrealProvider()" in registry, "technology lives in providers, registered in one place"
