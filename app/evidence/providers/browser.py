"""Optional browser/UI provider - entirely behind the capability registry (the scheduler knows nothing about
browsers). Disabled unless the provider setting browser.enabled=true.

Pages are DECLARED in the project's AI Studio manifest (.aistudio/evidence.json):
    "browser": {"pages": [{"name": "home", "url": "http://127.0.0.1:5173/", "runtime": "app"},
                          {"name": "static", "path": "web/index.html"}]}
Only local URLs (localhost / 127.0.0.1 / ::1) or repository files are allowed; a page whose runtime is
declared requires that runtime as a resource. Nothing else can be browsed.

Capabilities per page (keys browser:<page>:...):
    local_page_probe     load it; HTTP status and optional literal text in the page
    console_log_capture  load it; bounded console messages, checked for literal markers
    visual_capture       load it; a screenshot (an image artifact - captured, not judged)
    ui_interaction       load it and run a SHORT, validated action list (args):
                         goto:<path> | click:<selector> | press:<key> | type:<selector>=<text> |
                         wait:<ms> | expect:<text> | expect:<selector>=<text>
                         typing into password fields is refused; text and selectors are bounded
A browser driver must genuinely exist: the Playwright driver is used only when the package and a browser
are installed (never installed by AI Studio); otherwise every capability is UNAVAILABLE with the reason.
Pages and the browser are always closed after a step.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.evidence import readiness as rd
from app.evidence.model import FAILED, INFERRED, PASSED, UNAVAILABLE, Capability, PlanStep
from app.evidence.providers.base import CapabilityProvider, DiscoveryContext, ExecContext, StepOutcome
from app.evidence.providers.command import MANIFEST, declared_provenance

MAX_PAGES = 8
MAX_ACTIONS = 8
MAX_CONSOLE = 200
_SELECTOR = re.compile(r"^[A-Za-z0-9_#.\-\[\]=:>*()+~^$| ]{1,150}$")
_KEY = re.compile(r"^[A-Za-z0-9+]{1,30}$")
_TEXT = re.compile(r"^[A-Za-z0-9 _.,:;!?@#()/+-]{1,100}$")
_PATH = re.compile(r"^/[A-Za-z0-9_./?=&%-]{0,200}$")


class BrowserSession(ABC):
    @abstractmethod
    async def goto(self, url: str, timeout: float) -> int | None: ...

    @abstractmethod
    async def click(self, selector: str, timeout: float) -> None: ...

    @abstractmethod
    async def press(self, key: str) -> None: ...

    @abstractmethod
    async def type(self, selector: str, text: str, timeout: float) -> None:
        """Must refuse password inputs."""

    @abstractmethod
    async def text(self, selector: str = "") -> str: ...

    @abstractmethod
    async def screenshot(self) -> tuple[bytes, int, int]: ...

    @abstractmethod
    def console(self) -> list[str]: ...

    @abstractmethod
    async def close(self) -> None: ...


class BrowserDriver(ABC):
    name = "driver"

    @abstractmethod
    def available(self) -> tuple[bool, str]: ...

    @abstractmethod
    async def open(self, timeout: float) -> BrowserSession: ...


class PlaywrightDriver(BrowserDriver):
    """Uses an EXISTING Playwright installation (package + a downloaded browser); never installs anything."""
    name = "playwright"

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("playwright") is None:
            return False, "Playwright is not installed (AI Studio never installs it)"
        base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or os.path.join(
            os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache"), "ms-playwright")
        try:
            if not any(n.startswith("chromium") for n in os.listdir(base)):
                return False, "Playwright is installed but no Chromium browser was downloaded"
        except OSError:
            return False, "Playwright is installed but no browser was downloaded"
        return True, "Playwright with a downloaded Chromium (not verified until used)"

    async def open(self, timeout: float) -> BrowserSession:
        from playwright.async_api import async_playwright  # type: ignore
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True, timeout=timeout * 1000)
        page = await browser.new_page()
        return _PlaywrightSession(pw, browser, page)


class _PlaywrightSession(BrowserSession):
    def __init__(self, pw: Any, browser: Any, page: Any):
        self.pw, self.browser, self.page = pw, browser, page
        self._console: list[str] = []
        page.on("console", lambda m: self._console.append(f"{m.type}: {m.text}"[:500])
                if len(self._console) < MAX_CONSOLE else None)
        page.on("pageerror", lambda e: self._console.append(f"pageerror: {e}"[:500]))

    async def goto(self, url: str, timeout: float) -> int | None:
        resp = await self.page.goto(url, timeout=timeout * 1000)
        return resp.status if resp is not None else None

    async def click(self, selector: str, timeout: float) -> None:
        await self.page.click(selector, timeout=timeout * 1000)

    async def press(self, key: str) -> None:
        await self.page.keyboard.press(key)

    async def type(self, selector: str, text: str, timeout: float) -> None:
        el = await self.page.wait_for_selector(selector, timeout=timeout * 1000)
        if el is None or (await el.get_attribute("type") or "").lower() == "password":
            raise PermissionError("typing into password fields is not allowed")
        await el.fill(text)

    async def text(self, selector: str = "") -> str:
        return (await self.page.inner_text(selector or "body"))[:65536]

    async def screenshot(self) -> tuple[bytes, int, int]:
        data = await self.page.screenshot(type="png")
        vp = self.page.viewport_size or {"width": 0, "height": 0}
        return data, int(vp.get("width") or 0), int(vp.get("height") or 0)

    def console(self) -> list[str]:
        return list(self._console)

    async def close(self) -> None:
        try:
            await self.browser.close()
        finally:
            await self.pw.stop()


def parse_action(a: str) -> tuple[str, str, str] | None:
    """(verb, target, value) of one validated action, or None."""
    verb, _, rest = str(a).partition(":")
    if verb == "goto" and _PATH.match(rest):
        return verb, rest, ""
    if verb == "click" and _SELECTOR.match(rest):
        return verb, rest, ""
    if verb == "press" and _KEY.match(rest):
        return verb, rest, ""
    if verb == "wait" and rest.isdigit() and 0 < int(rest) <= 5000:
        return verb, rest, ""
    if verb in ("type", "expect") and "=" in rest:
        sel, _, text = rest.partition("=")
        if _SELECTOR.match(sel) and _TEXT.match(text):
            return verb, sel, text
    if verb == "expect" and _TEXT.match(rest):
        return verb, "", rest
    return None


class BrowserProvider(CapabilityProvider):
    name = "browser"
    version = "1"

    def __init__(self, driver: BrowserDriver | None = None):
        self.driver = driver or PlaywrightDriver()
        self.last_notes: list[str] = []

    async def discover(self, ctx: DiscoveryContext) -> list[Capability]:
        self.last_notes = []
        if ctx.provider_setting("browser.enabled").lower() not in ("1", "true", "yes", "on"):
            return []  # optional provider: off unless enabled
        path = os.path.join(ctx.repo_path, *MANIFEST.split("/"))
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f) if os.path.getsize(path) <= 64 * 1024 else {}
        except (OSError, ValueError):
            return []
        pages = ((data or {}).get("browser") or {}).get("pages") if isinstance(data, dict) else None
        if not isinstance(pages, list):
            return []
        ok, why = self.driver.available()
        prov = declared_provenance(ctx, MANIFEST)
        caps: list[Capability] = []
        for d in pages[:MAX_PAGES]:
            if not isinstance(d, dict) or not str(d.get("name") or "").strip():
                continue
            name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(d["name"]).strip())[:40]
            url, bad = "", ""
            if d.get("url"):
                url = str(d["url"]).strip()
                bad = rd.check_local_url(url)
            elif d.get("path"):
                rel = str(d["path"]).replace("\\", "/")
                full = os.path.abspath(os.path.join(ctx.repo_path, *rel.split("/")))
                if not rd.relative_path_ok(rel) or not full.startswith(os.path.abspath(ctx.repo_path) + os.sep):
                    bad = "path must be a repository-relative file"
                else:
                    url = Path(full).as_uri()
            else:
                bad = "a page needs a local url or a repository path"
            need = f"runtime:{re.sub(r'[^A-Za-z0-9_.-]+', '-', str(d.get('runtime')).strip())}" \
                if d.get("runtime") else ""
            reason = bad or ("" if ok else why) or f"declared page; {why}"
            for suffix, kind, label, desc, args in (
                    ("probe", "local_page_probe", "load", "loads the page; HTTP status and optional literal text "
                                                          "(expect_output)", False),
                    ("console", "console_log_capture", "console", "loads the page and captures its console messages "
                                                                   "(fail_output / expect_output markers)", False),
                    ("screenshot", "visual_capture", "screenshot", "loads the page and captures a screenshot "
                                                                    "(captured only - not judged)", False),
                    ("ui", "ui_interaction", "interact", "loads the page and runs a short action list (args): "
                                                         "goto:/path click:<sel> press:<key> type:<sel>=<text> "
                                                         "wait:<ms> expect:<text> expect:<sel>=<text>", True)):
                constraints: dict[str, Any] = {"timeout_seconds": 60, "accepts_args": args}
                if args:
                    constraints.update(max_args=MAX_ACTIONS, args_help="the action list")
                if need:
                    constraints["requires_resource"] = need
                caps.append(Capability(
                    id="", kind=kind, name=f"browser {label}: {name}", description=desc, source="repository",
                    confidence=INFERRED if not bad and ok else UNAVAILABLE, provider=self.name,
                    key=f"browser:{name}:{suffix}", constraints=constraints,
                    details={"url": url, "page": name, "driver": self.driver.name}, reason=reason,
                    provenance=dict(prov, source_detail="declared browser page")))
        return caps

    def validate(self, cap: Capability, step: PlanStep) -> str:
        if cap.kind != "ui_interaction":
            return "" if not step.args else f"{cap.id} does not accept arguments"
        if not step.args or len(step.args) > MAX_ACTIONS:
            return f"{cap.id} takes 1-{MAX_ACTIONS} actions"
        bad = [a for a in step.args if parse_action(a) is None]
        return f"invalid action(s): {', '.join(repr(x) for x in bad[:3])}" if bad else ""

    async def execute(self, cap: Capability, step: PlanStep, ctx: ExecContext) -> StepOutcome:
        why = self.validate(cap, step)
        if why:
            return StepOutcome(UNAVAILABLE, reason=why, cause="strategy")
        ok, reason = self.driver.available()
        if not ok:
            return StepOutcome(UNAVAILABLE, reason=reason, cause="strategy")
        url = str(cap.details.get("url") or "")
        if url.startswith("http") and rd.check_local_url(url):
            return StepOutcome(UNAVAILABLE, reason="not a local URL", cause="strategy")
        t0 = time.monotonic()
        try:
            session = await self.driver.open(min(ctx.timeout, 30.0))
        except Exception as exc:
            return StepOutcome(UNAVAILABLE, reason=f"the browser could not start: {exc}"[:300], cause="strategy")
        try:
            try:
                status = await session.goto(url, min(ctx.timeout, 30.0))
            except Exception as exc:
                return StepOutcome(FAILED, summary=f"the page {cap.details.get('page')} did not load: {exc}"[:300],
                                   reason="page did not load", duration=time.monotonic() - t0)
            if status is not None and status >= 400:
                return StepOutcome(FAILED, summary=f"the page {cap.details.get('page')} answered HTTP {status}",
                                   reason=f"HTTP {status}", duration=time.monotonic() - t0)
            return await self._run(cap, step, ctx, session, url, t0)
        finally:
            try:
                await session.close()
            except Exception:
                pass

    async def _run(self, cap: Capability, step: PlanStep, ctx: ExecContext, session: BrowserSession, url: str,
                   t0: float) -> StepOutcome:
        page = cap.details.get("page")
        if cap.kind == "visual_capture":
            data, w, h = await session.screenshot()
            return StepOutcome(PASSED, summary=f"captured a {w}x{h} screenshot of {page} - captured only, not judged",
                               duration=time.monotonic() - t0, artifact_kind="image",
                               files=[{"data": data, "media_type": "image/png", "kind": "image",
                                       "label": f"{step.key}-screenshot", "description": f"screenshot of {page}",
                                       "metadata": {"width": w, "height": h, "page": page}}])
        if cap.kind == "console_log_capture":
            lines = session.console()
            text = "\n".join(lines)
            if step.fail_output and step.fail_output in text:
                return StepOutcome(FAILED, summary=f"{page}: console shows {step.fail_output!r}", output=text,
                                   reason="failure marker found", duration=time.monotonic() - t0)
            if step.expect_output and step.expect_output not in text:
                return StepOutcome(FAILED, summary=f"{page}: console lacks {step.expect_output!r}", output=text,
                                   reason="expected output missing", duration=time.monotonic() - t0)
            return StepOutcome(PASSED, summary=f"{page}: {len(lines)} console message(s)", output=text,
                               duration=time.monotonic() - t0)
        body = await session.text()
        if cap.kind == "local_page_probe":
            if step.expect_output and step.expect_output not in body:
                return StepOutcome(FAILED, summary=f"{page} loaded but {step.expect_output!r} is missing",
                                   output=body[:8000], reason="expected output missing", duration=time.monotonic() - t0)
            return StepOutcome(PASSED, summary=f"{page} loaded" + (f" and shows {step.expect_output!r}"
                                                                   if step.expect_output else ""),
                               output=body[:8000], duration=time.monotonic() - t0)
        log = []
        for a in step.args:  # ui_interaction: the validated action list, in order
            verb, target, value = parse_action(a) or ("", "", "")
            try:
                if verb == "goto":
                    base = url.split("/", 3)
                    await session.goto("/".join(base[:3]) + target if url.startswith("http") else url, 30.0)
                elif verb == "click":
                    await session.click(target, 10.0)
                elif verb == "press":
                    await session.press(target)
                elif verb == "type":
                    await session.type(target, value, 10.0)
                elif verb == "wait":
                    await asyncio.sleep(int(target) / 1000)
                elif verb == "expect":
                    found = await session.text(target) if target else await session.text()
                    if value not in found:
                        log.append(f"{a}: NOT FOUND")
                        return StepOutcome(FAILED, summary=f"{page}: expected {value!r}"
                                                           + (f" in {target}" if target else "") + " was not shown",
                                           output="\n".join(log), reason="expectation failed",
                                           duration=time.monotonic() - t0)
                log.append(f"{a}: ok")
            except PermissionError as exc:
                return StepOutcome(UNAVAILABLE, summary=str(exc), reason=str(exc), cause="strategy",
                                   output="\n".join(log))
            except Exception as exc:
                log.append(f"{a}: {exc}"[:300])
                return StepOutcome(FAILED, summary=f"{page}: action {a!r} failed: {exc}"[:300],
                                   output="\n".join(log), reason="action failed", duration=time.monotonic() - t0)
        return StepOutcome(PASSED, summary=f"{page}: {len(step.args)} action(s) succeeded", output="\n".join(log),
                           duration=time.monotonic() - t0)
