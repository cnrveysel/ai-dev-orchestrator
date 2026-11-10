"""Pipeline V2 bounded repair: deterministic evidence handling (no model involved).

  failure fingerprint  normalized failing assertions/tests of a verification run: ANSI
                       colors, timestamps, absolute/temporary paths, file:line positions,
                       addresses and incidental measurements are removed, so the SAME
                       underlying failures produce the SAME fingerprint across rounds
  score                build failure > critical failure (timeout/crash) > number of failing
                       assertions > number of failing commands; lower is better
  path rules           which files are acceptance tests (protected during repair) and which
                       new files are scratch/probe files (counted as stuck evidence)
  guard                the limits that stop a repair session
  diagnosis packet     the compact, byte-bounded evidence sent with DIAGNOSE_BLOCKER

Progress is only ever measured from verification results, never from the worker's prose.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any

from app.integrations import path_filter
from app.models.attempt import VerificationResult
from app.models.phase import Phase
from app.models.repair import RepairGuardState
from app.orchestration.reviewer import _cut, utf8_len

MAX_SIGNATURES = 40
MAX_LINES_KEPT = 24
MAX_PROTECTED = 200
MAX_EXACT_BLOBS = 500  # changed/untracked files kept byte-exact per best-known record
# A second restoration of the best-known state in one repair session means the worker keeps
# regressing: the session is stopped and the blocker escalated.
MAX_RESTORES_PER_SESSION = 2

# ------------------------------------------------------------ failure fingerprinting

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
_ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
_PATH_CHARS = r"[^\s\\/:*?\"'<>|()\[\],;]"
_WIN_PATH_RE = re.compile(rf"(?<![\w])[A-Za-z]:[\\/](?:{_PATH_CHARS}+[\\/])*({_PATH_CHARS}*)")
_POSIX_PATH_RE = re.compile(r"(?<![\w.~])/(?:[\w.@+-]+/)+([\w.@+-]*)")
_FILE_POS_RE = re.compile(r"(\.[A-Za-z][A-Za-z0-9]{0,5}):\d+(?::\d+)?")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_LONG_ID_RE = re.compile(r"\b(?=[0-9a-fA-F-]*\d)(?=[0-9a-fA-F-]*[a-fA-F])[0-9a-fA-F]{8,}(?:-[0-9a-fA-F]{4,})*\b")
_DECIMAL_RE = re.compile(r"(?<![\w.])\d+\.\d+")
_MEASURE_RE = re.compile(r"\b(took|in|after|elapsed|duration|time|got|actual|received|was|seed|pid|port|tick|frame|"
                         r"iteration|attempt|at)(\s*[:=]?\s*)-?\d+", re.IGNORECASE)
_GROUP_RE = re.compile(r"\(([^()]*\d[^()]*)\)|\[([^\[\]]*\d[^\[\]]*)\]")
_NUM_RUN_RE = re.compile(r"(?:<n>\s*){2,}")
_WS_RE = re.compile(r"\s+")
_LEAD_TS_RE = re.compile(r"^\s*[\[(]?(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
                         r"|\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?)[\])]?\s*")

# A failing assertion/test line (case-sensitive markers, as test runners print them).
_PRIMARY_RE = re.compile(
    r"^\s*(?:[-*•>|]\s*)?(?:FAIL(?:ED|URE|S)?\b|not ok\b|[✗✘×✖❌]|●\s|ERROR\b)"
    r"|\berror\s+[A-Z]{1,4}\d{2,5}\b"  # error TS2345 / C2065 / CS0103
    r"|:\s*(?:fatal )?error:\s")  # gcc/clang: file:line: error:
# Used only when a failed command printed no primary marker at all.
_SECONDARY_RE = re.compile(r"\b[A-Z]\w*(?:Error|Exception)\b[:!]|\bassert(?:ion)? failed\b|\bpanicked at\b",
                           re.IGNORECASE)
_SUMMARY_RE = re.compile(r"^\s*FAILED\s*\((?:failures|errors)=|^\s*[=\-_*]{3,}.*[=\-_*]{3,}\s*$|^\s*ERROR\s*:?\s*$")
_CRASH_RE = re.compile(r"Segmentation fault|segfault|Access violation|0xC0000005|Unhandled exception|FATAL ERROR|"
                       r"heap out of memory|out of memory|stack overflow|Maximum call stack|panicked at", re.IGNORECASE)
_COMPILE_RE = re.compile(r"\berror\s+(?:TS|C|CS)\d{2,5}\b|\bSyntaxError\b|Cannot find module|Module not found|"
                         r"Could not resolve|\bBUILD FAILED\b|\bBuild failed\b|Compilation failed|✘ \[ERROR\]|"
                         r":\s*(?:fatal )?error:\s")
_BUILD_CMD_RE = re.compile(r"\b(?:build|compile|tsc|esbuild|webpack|rollup|msbuild|cmake|make|ninja|cargo\s+build|"
                           r"dotnet\s+build|gradlew?|mvn|javac|UnrealBuildTool|RunUAT|Build\.bat|typecheck|"
                           r"type-check)\b", re.IGNORECASE)
_TEST_FAIL_MARK_RE = re.compile(r"^\s*(?:[-*•>|]\s*)?(?:FAIL(?:ED|URE|S)?\b|not ok\b|[✗✘×✖❌]|●\s)")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def normalize_failure_line(line: str) -> str:
    """Stable identity of one failure line: unstable noise removed, assertion text kept."""
    s = _LEAD_TS_RE.sub("", strip_ansi(line)).strip()  # a log-line timestamp prefix is never identity
    s = _ISO_TS_RE.sub("<ts>", s)
    s = _CLOCK_RE.sub("<ts>", s)
    s = _WIN_PATH_RE.sub(lambda m: "<abs>/" + m.group(1), s)
    s = _POSIX_PATH_RE.sub(lambda m: "<abs>/" + m.group(1), s)
    s = _FILE_POS_RE.sub(r"\1:<n>", s)
    s = _HEX_RE.sub("<hex>", s)
    s = _LONG_ID_RE.sub("<id>", s)
    s = _DECIMAL_RE.sub("<n>", s)
    s = _MEASURE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}<n>", s)
    s = _GROUP_RE.sub(lambda m: "(" + re.sub(r"\d+", "<n>", m.group(1)) + ")" if m.group(1) is not None
                      else "[" + re.sub(r"\d+", "<n>", m.group(2)) + "]", s)
    s = _NUM_RUN_RE.sub("<n> ", s)
    if s.startswith("FAILED ") and " - " in s:  # pytest summary: the test id is the identity
        s = s.split(" - ", 1)[0]
    s = re.sub(r"^(\s*not ok)\s+\d+", r"\1", s)  # TAP numbering changes when tests are added
    return _WS_RE.sub(" ", s).strip()[:240]


def failure_lines(output: str) -> tuple[list[str], list[str]]:
    """(raw lines, normalized signatures) of the failures printed by one command."""
    lines = [_LEAD_TS_RE.sub("", strip_ansi(x)).rstrip() for x in (output or "").splitlines()]
    picked = [x for x in lines if x.strip() and _PRIMARY_RE.search(x) and not _SUMMARY_RE.search(x)]
    if not picked:
        picked = [x for x in lines if x.strip() and _SECONDARY_RE.search(x)]
    raw: list[str] = []
    sigs: list[str] = []
    for x in picked:
        sig = normalize_failure_line(x)
        if sig and sig not in sigs:
            sigs.append(sig)
            raw.append(x.strip()[:300])
    return raw, sigs


def _command_label(cmd: str) -> str:
    return normalize_failure_line(cmd)[:80]


@dataclass(frozen=True)
class VerificationScore:
    commands: int = 0  # commands actually executed
    failed_commands: int = 0
    failing: int = 0  # distinct failing assertions/tests (at least 1 per failed command)
    build_failed: bool = False
    critical: bool = False  # timeout / crash

    def key(self) -> tuple[int, int, int, int]:
        return int(self.build_failed), int(self.critical), self.failing, self.failed_commands

    def better_than(self, other: "VerificationScore") -> bool:
        return self.key() < other.key()

    def describe(self) -> str:
        if not self.commands:
            return "no verification evidence"
        if not self.failed_commands:
            return f"clean ({self.commands} command(s) passed)"
        parts = [f"{self.failing} failing assertion(s)/test(s) in {self.failed_commands} command(s)"]
        if self.build_failed:
            parts.insert(0, "BUILD FAILED")
        if self.critical:
            parts.insert(0, "CRITICAL (timeout/crash)")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"commands": self.commands, "failed_commands": self.failed_commands, "failing": self.failing,
                "build_failed": self.build_failed, "critical": self.critical}

    @classmethod
    def from_dict(cls, d: Any) -> "VerificationScore":
        d = d if isinstance(d, dict) else {}
        return cls(int(d.get("commands") or 0), int(d.get("failed_commands") or 0), int(d.get("failing") or 0),
                   bool(d.get("build_failed")), bool(d.get("critical")))


@dataclass
class VerificationEvidence:
    score: VerificationScore = field(default_factory=VerificationScore)
    fingerprint: str = ""  # "" when clean or when nothing ran
    failures: list[str] = field(default_factory=list)  # normalized signatures, sorted
    lines: list[str] = field(default_factory=list)  # the raw failure lines (for prompts/diagnosis)

    @property
    def has_evidence(self) -> bool:
        return self.score.commands > 0

    @property
    def passed(self) -> bool:
        return self.has_evidence and self.score.failed_commands == 0

    def summary(self) -> str:
        return self.score.describe()

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score.to_dict(), "fingerprint": self.fingerprint, "failures": list(self.failures),
                "lines": list(self.lines)}

    @classmethod
    def from_dict(cls, d: Any) -> "VerificationEvidence | None":
        if not isinstance(d, dict) or "score" not in d:
            return None
        return cls(VerificationScore.from_dict(d.get("score")), str(d.get("fingerprint") or ""),
                   [str(x) for x in d.get("failures") or []], [str(x) for x in d.get("lines") or []])


def evaluate_verification(results: list[VerificationResult]) -> VerificationEvidence:
    """Score and fingerprint one verification run (skipped commands are no evidence)."""
    ran = [v for v in results if not v.skipped_reason]
    failed = [v for v in ran if not v.passed]
    sigs: list[str] = []
    raw: list[str] = []
    build = critical = False
    failing = 0
    for v in failed:
        label = _command_label(v.command)
        out = v.output_tail or ""
        lines, cmd_sigs = failure_lines(out)
        if v.exit_code is None:  # the verifier timed out
            critical = True
            cmd_sigs = cmd_sigs + ["TIMEOUT"]
            lines = lines + [f"{v.command}: TIMEOUT"]
        if not cmd_sigs:
            last = [x for x in strip_ansi(out).splitlines() if x.strip()][-1:] or [""]
            cmd_sigs = [f"exit {v.exit_code}: {normalize_failure_line(last[0])}".rstrip(": ")]
            lines = [f"{v.command}: exit {v.exit_code}" + (f": {last[0].strip()[:200]}" if last[0].strip() else "")]
        if _CRASH_RE.search(out):
            critical = True
        if _COMPILE_RE.search(out) or (_BUILD_CMD_RE.search(v.command) and not any(
                _TEST_FAIL_MARK_RE.search(x) for x in strip_ansi(out).splitlines())):
            build = True
        failing += len([s for s in cmd_sigs if s != "TIMEOUT"]) or 1
        sigs += [f"{label} :: {s}" for s in cmd_sigs]
        raw += lines
    sigs = sorted(set(sigs))[:MAX_SIGNATURES]
    fp = hashlib.sha1("\n".join(sigs).encode("utf-8")).hexdigest()[:16] if sigs else ""
    score = VerificationScore(len(ran), len(failed), failing if failed else 0, build, critical)
    return VerificationEvidence(score, fp, sigs, raw[:MAX_LINES_KEPT])


def trend(prev: VerificationEvidence | None, cur: VerificationEvidence) -> str:
    """first | same | improved | worse | changed (a different failure set with the same score)."""
    if prev is None or not prev.has_evidence:
        return "first"
    if cur.fingerprint == prev.fingerprint:
        return "same"
    if cur.score.better_than(prev.score):
        return "improved"
    if prev.score.better_than(cur.score):
        return "worse"
    return "changed"


def fixed_and_new(prev: VerificationEvidence | None, cur: VerificationEvidence) -> tuple[list[str], list[str]]:
    before = set(prev.failures) if prev else set()
    now = set(cur.failures)
    return sorted(before - now), sorted(now - before)


# ------------------------------------------------------------ guard

def _limit(v: Any, minimum: int = 1) -> int:
    try:
        return max(minimum, int(v))
    except (TypeError, ValueError):
        return minimum


def guard_trip(g: RepairGuardState, o: Any) -> str:
    """Why the current repair session must stop ("" = within all limits)."""
    wall, tools = _limit(o.repair_max_wall_seconds), _limit(o.repair_max_tool_calls)
    same, probes = _limit(o.repair_max_same_failure_rounds), _limit(o.repair_max_new_probe_files, 0)
    no_imp = _limit(getattr(o, "repair_max_no_improvement_rounds", 4))
    if g.wall_used >= wall:
        return f"wall-clock budget reached ({g.wall_used:.0f}s/{wall}s)"
    if g.tool_calls >= tools:
        return f"tool budget reached {g.tool_calls}/{tools}"
    if g.same_rounds >= same:
        return f"failure fingerprint unchanged for {g.same_rounds}/{same} verification rounds"
    if g.no_improvement_rounds >= no_imp:
        return (f"no verification improvement over the best-known state for {g.no_improvement_rounds}/{no_imp} "
                "rounds")
    if len(g.probe_files) > probes:
        return f"{len(g.probe_files)} new probe/debug files without verification progress (max {probes})"
    if g.restores >= MAX_RESTORES_PER_SESSION:
        return f"repeated regression from the best-known state ({g.restores} restorations)"
    return ""


# ------------------------------------------------------------ path rules

TEST_DIR_NAMES = frozenset({"test", "tests", "__tests__", "spec", "specs", "e2e", "acceptance", "testing"})
_TEST_NAME_RE = re.compile(r"(^test_.+\.\w+$)|(_test\.\w+$)|(\.(test|spec)\.\w+$)|(_spec\.\w+$)|"
                           r"(Tests?\.(java|cs|kt|swift|cpp)$)|((^|[._-])(smoke|acceptance)([._-]|$))", re.IGNORECASE)
PROBE_DIR_NAMES = frozenset({"scratch_smoke", "scratch", "tmp", "temp", ".tmp", "probe", "probes", "debug", "diag",
                             "diagnostics"})
PROBE_TOKENS = frozenset({"probe", "scratch", "repro", "debug", "dbg", "tmp", "temp", "diag", "diagnose", "diagnostic",
                          "investigate", "experiment"})


def norm_path(p: str) -> str:
    p = str(p or "").strip().strip("\"'").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def _parts(p: str) -> list[str]:
    return [x for x in norm_path(p).split("/") if x]


def is_test_path(p: str) -> bool:
    parts = _parts(p)
    if not parts:
        return False
    return any(d.lower() in TEST_DIR_NAMES for d in parts[:-1]) or bool(_TEST_NAME_RE.search(parts[-1]))


def is_probe_path(p: str) -> bool:
    """Scratch/probe/debug files by location or name ("probe2.js", "debug_deadlock.ts",
    "scratch_smoke/x.mjs"). A regular test ("tests/test_x.py") is never a probe."""
    parts = _parts(p)
    if not parts:
        return False
    if any(d.lower() in PROBE_DIR_NAMES for d in parts[:-1]):
        return True
    stem = os.path.splitext(parts[-1])[0]
    tokens = re.split(r"[^a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", stem).lower())
    return any(t in PROBE_TOKENS for t in tokens)


def is_authorized(path: str, patterns: list[str]) -> bool:
    p = norm_path(path).lower()
    for raw in patterns or []:
        pat = norm_path(raw).lower()
        if not pat:
            continue
        if pat in ("*", "**", "all", "any") or p == pat or fnmatch.fnmatchcase(p, pat) \
                or p.endswith("/" + pat) or fnmatch.fnmatchcase(p, "*/" + pat):
            return True
    return False


_SHELL_SPLIT_RE = re.compile(r"&&|\|\||[;|<>]")


def _command_tokens(cmd: str) -> list[str]:
    toks: list[str] = []
    for part in _SHELL_SPLIT_RE.split(cmd or ""):
        for tok in re.findall(r"\"[^\"]*\"|'[^']*'|\S+", part):
            tok = tok.strip("\"'")
            if tok.startswith("-"):
                if "=" not in tok:
                    continue
                tok = tok.split("=", 1)[1]
            tok = tok.split("::", 1)[0]  # pytest node ids
            if tok:
                toks.append(tok)
    return toks


def _repo_rel(tok: str, repo: str) -> str:
    t = tok.replace("\\", "/")
    if os.path.isabs(t):
        try:
            rel = os.path.relpath(t, repo)
        except ValueError:  # another drive
            return ""
        t = rel.replace("\\", "/")
    t = norm_path(t)
    if not t or t.startswith("../") or t == ".." or t == ".":
        return ""
    return t


def acceptance_files(commands: list[str], repo: str, phase_paths: list[str], likely: list[str]) -> list[str]:
    """The acceptance/verifier files of a phase: files named by its verification commands (tests
    under a named directory), plus existing test files the phase created/changed or planned."""
    found: list[str] = []

    def add(rel: str) -> None:
        rel = norm_path(rel)
        if rel and rel not in found and len(found) < MAX_PROTECTED:
            found.append(rel)

    for cmd in commands or []:
        for tok in _command_tokens(cmd):
            rel = _repo_rel(tok, repo)
            if not rel:
                continue
            full = os.path.join(repo, rel)
            if os.path.isfile(full):
                add(rel)
            elif os.path.isdir(full) and not path_filter.generated_dir(rel + "/x"):
                for root, dirs, files in os.walk(full):
                    dirs[:] = [d for d in dirs if d.lower() not in {x.lower() for x in path_filter.GENERATED_DIRS}
                               and not d.startswith(".")]
                    for f in sorted(files):
                        r = norm_path(os.path.relpath(os.path.join(root, f), repo))
                        if is_test_path(r) and not is_probe_path(r):
                            add(r)
    for rel in list(phase_paths or []) + list(likely or []):
        rel = norm_path(rel)
        if rel and is_test_path(rel) and not is_probe_path(rel) and os.path.isfile(os.path.join(repo, rel)):
            add(rel)
    return found


# ------------------------------------------------------------ DIAGNOSE_BLOCKER packet

@dataclass
class DiagnosisPacket:
    phase: Phase
    reason: str  # why the repair guard stopped
    repo_path: str = ""
    goal: str = ""
    checkpoint: dict[str, Any] = field(default_factory=dict)
    current: VerificationEvidence | None = None
    best: VerificationEvidence | None = None
    changed_files: list[str] = field(default_factory=list)
    diff: str = ""
    rounds: list[dict[str, Any]] = field(default_factory=list)
    previous_diagnoses: list[dict[str, Any]] = field(default_factory=list)
    protected_tests: list[str] = field(default_factory=list)
    user_guidance: str = ""


@dataclass(frozen=True)
class DiagnosisLimits:
    max_prompt_bytes: int = 30000
    max_diff_bytes: int = 12000

    @classmethod
    def from_settings(cls, s: Any) -> "DiagnosisLimits":
        prompt = max(4000, int(getattr(s, "diagnose_max_prompt_bytes", 30000)))
        return cls(prompt, max(0, min(int(getattr(s, "diagnose_max_diff_bytes", 12000)), prompt // 2)))


def _evidence_lines(ev: VerificationEvidence | None, n: int) -> list[str]:
    if ev is None or not ev.has_evidence:
        return ["(no verification evidence)"]
    if ev.passed:
        return ["(verification passed)"]
    return [ev.summary()] + [f"- {x}" for x in (ev.lines or ev.failures)[:n]]


def build_diagnosis_prompt(pk: DiagnosisPacket, limits: DiagnosisLimits) -> tuple[str, dict[str, int]]:
    p = pk.phase
    head = [
        "DIAGNOSE_BLOCKER - a bounded repair of this phase is stuck.",
        "Do NOT edit files. Give the worker precise, structured guidance, or return BLOCKED if progress needs a user "
        "decision, missing information or an environment fix.",
        f"Guard reason: {pk.reason}",
        "", f"PHASE {p.id}: {p.title}", "OBJECTIVE", _cut(p.description or "(none)", 1500),
        "", "ACCEPTANCE CRITERIA", *[f"{i}. {c}" for i, c in enumerate(p.acceptance_criteria, 1)],
    ]
    if p.human_criteria:  # V4: derived from the user's verification; kept separate from the criteria above
        head += ["", "HUMAN-DERIVED ACCEPTANCE CRITERIA", *[f"- {c.id}: {c.text}" for c in p.human_criteria]]
    if p.verification_commands:
        head += ["", "VERIFICATION COMMANDS (run by the orchestrator)", *[f"- {c}" for c in p.verification_commands]]
    done = [f"{s.id} {s.title}" for s in p.subtasks if s.done]
    head += ["", "COMPLETED SUBTASKS: " + (", ".join(done) or "(none reported)")]
    ck = pk.checkpoint or {}
    head += ["", "CHECKPOINT FEEDBACK THAT STARTED THE REPAIR", f"summary: {_cut(str(ck.get('summary', '')), 800)}"]
    head += [f"- problem: {x}" for x in (ck.get("problems") or [])[:8]]
    head += [f"- fix: {x}" for x in (ck.get("corrective_instructions") or [])[:8]]
    fb = ck.get("human_feedback") or {}
    if fb:
        head += [f"- the user's verification: {fb.get('verdict')}" + (f" - \"{_cut(str(fb.get('note', '')), 800)}\""
                                                                      if fb.get("note") else "")]
    head += [f"- keep (confirmed working): {x}" for x in (ck.get("do_not_redo") or [])[:6]]
    if pk.user_guidance:
        head += ["", "USER GUIDANCE", _cut(pk.user_guidance, 800)]
    head += ["", "CURRENT FAILING ASSERTIONS", *_evidence_lines(pk.current, 16)]
    head += ["", "BEST-KNOWN STATE IN THIS REPAIR", *_evidence_lines(pk.best, 10)]
    if pk.protected_tests:
        head += ["", "PROTECTED ACCEPTANCE TESTS (the worker may not modify them unless you authorize it)",
                 ", ".join(pk.protected_tests[:30])]
    mandatory = "\n".join(head)

    rounds = ["RECENT REPAIR ROUNDS (oldest first)"]
    for r in pk.rounds[-8:]:
        rounds.append(f"- round {r.get('n')}: {r.get('trend', '')} - {r.get('summary', '')}"
                      + (f"; {r.get('note')}" if r.get("note") else ""))
        if r.get("worker"):
            rounds.append(f"  worker said: {_cut(str(r['worker']), 400)}")
    prev = ["EARLIER DIAGNOSES (do not repeat advice that did not work)"]
    for d in pk.previous_diagnoses[-3:]:
        prev.append(f"- {d.get('reason', '')}: {_cut(str(d.get('diagnosis', '')), 500)}")
        prev += [f"  action: {x}" for x in (d.get("recommended_actions") or [])[:4]]
    files = "CHANGED FILES IN THIS REPAIR\n" + (", ".join(pk.changed_files[:40]) or "(none)")
    decide = ("ANSWER with the structured schema: status REPAIR with diagnosis, recommended_actions (concrete, "
              "ordered), avoid (approaches already tried that failed or regressed), target_files and target_tests "
              "(fast targeted checks to run after each change); or status BLOCKED with blocked_reason. Only if an "
              "existing acceptance test is itself wrong, list it in authorized_test_changes.")

    def compose(opt_bytes: int, diff_bytes: int) -> str:
        parts = [mandatory]
        if len(rounds) > 1:
            parts.append(_cut("\n".join(rounds), opt_bytes))
        if len(prev) > 1:
            parts.append(_cut("\n".join(prev), opt_bytes // 2))
        parts.append(_cut(files, 2000))
        if pk.diff and diff_bytes > 0:
            d = _cut(pk.diff, diff_bytes)
            parts.append("RELEVANT DIFF (repair changes so far" + (", truncated)" if d != pk.diff else ")") + "\n" + d)
        parts.append(decide)
        return "\n\n".join(parts) + "\n"

    prompt = ""
    for opt, diff in ((4000, limits.max_diff_bytes), (1500, limits.max_diff_bytes // 2), (500, 0)):
        prompt = compose(opt, diff)
        if utf8_len(prompt) <= limits.max_prompt_bytes:
            break
    else:  # even the minimal packet is too large: shorten the mandatory part, keep the instructions
        mandatory = _cut(mandatory, max(1000, limits.max_prompt_bytes - utf8_len(decide) - 2500))
        prompt = _cut(compose(300, 0), limits.max_prompt_bytes)
    return prompt, {"prompt_bytes": utf8_len(prompt), "limit_bytes": limits.max_prompt_bytes,
                    "diff_original_bytes": utf8_len(pk.diff)}
