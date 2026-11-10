"""Pipeline V2.1: codebase-memory preflight (list_projects / index_repository parsing) and the V2
worker's codebase-memory guidance. The codebase-memory CLI is always a scripted fake - no test
talks to a live codebase-memory server."""
import asyncio
import json
import re

import pytest

from app.config.settings import ClaudeSettings, IntegrationSettings, OrchestrationSettings, Settings
from app.integrations import codebase_memory as cbm
from app.models.phase import parse_phases
from app.models.task import Task
from app.orchestration.providers.claude_code_cli import limits_for
from app.workers.base import CODEBASE_MEMORY_DEGRADED_LINE, CODEBASE_MEMORY_LINES, WorkerTaskSpec, build_worker_prompt
from app.workers.phase_prompts import PhaseWorkSpec, build_phase_prompt, build_turn_prompt
from app.workers.process_runner import ProcessResult

REPO = r"C:\Users\Example\Documents\SampleGame"
ROOT = "C:/Users/Example/Documents/SampleGame"
PROJECT = "C-Users-Example-Documents-SampleGame"
OTHER = ("C-Users-Example-Documents-Unreal-Projects-OtherGame", "C:/Users/Example/Documents/Unreal Projects/OtherGame")

# Real codebase-memory-mcp 0.11.0 text output: the root is quoted ONLY when it contains spaces.
LIST_TEXT = ("projects: 2  (cols: name root_path branch)\n"
             f"  {PROJECT} {ROOT} ai-orchestrator/run-11\n"
             f"  {OTHER[0]} \"{OTHER[1]}\" master\n"
             "total: 2\nreturned: 2\nhas_more: false\n")
LIST_TEXT_OTHER_ONLY = ("projects: 1  (cols: name root_path branch)\n"
                        f"  {OTHER[0]} \"{OTHER[1]}\" master\n"
                        "total: 1\nreturned: 1\nhas_more: false\n")
INDEX_STATUS_TEXT = (f"project: {PROJECT}\nnodes: 546\nedges: 1676\nstatus: ready\nroot_path: {ROOT}\n"
                     "parse_partial:\n  files: 0\n  count: 0\n  truncated: false\n")
ADR_HINT = ("Project indexed. Consider creating an Architecture Decision Record: explore the codebase with "
            "get_architecture(aspects=['all']), then use manage_adr(mode='update') to persist architectural "
            "insights across sessions.")
INDEXED = {"project": PROJECT, "nodes": 546, "edges": 1676, "adr_present": False, "adr_hint": ADR_HINT,
           "artifact_present": False, "status": "indexed"}
# codebase-memory-mcp 0.11.0 tool surface (`codebase-memory-mcp --help`).
TOOLS_0_11 = {"index_repository", "search_graph", "query_graph", "trace_path", "get_code_snippet", "get_file_outline",
              "get_graph_schema", "compare_graphs", "get_architecture", "search_code", "list_projects",
              "delete_project", "index_status", "check_index_coverage", "detect_changes", "manage_adr",
              "ingest_traces"}


def list_json(*projects, has_more=False, offset=0):
    return json.dumps({"projects": [{"name": n, "root_path": r, "branch": "main"} for n, r in projects],
                       "total": len(projects) + offset, "offset": offset, "limit": 500, "returned": len(projects),
                       "has_more": has_more})


def envelope(text, is_error=False):
    return json.dumps({"content": [{"type": "text", "text": text}], "isError": is_error})


class FakeCLI:
    """Scripted `codebase-memory-mcp cli ...`: list_projects answers `before` until index_repository
    ran, then `after`."""

    def __init__(self, before, after=None, index=(0, json.dumps(INDEXED), "")):
        self.before, self.after, self.index = before, before if after is None else after, index
        self.calls: list[list[str]] = []
        self.indexed = False

    def install(self, monkeypatch, tmp_path):
        fake = self

        class Runner:
            async def run(self, argv, *, timeout=None, **kw):
                fake.calls.append(list(argv[1:]))
                if argv[1] == "--version":
                    return ProcessResult(argv, 0, "codebase-memory-mcp 0.11.0\n", "", 0.0)
                tool = argv[3]
                if tool == "index_repository":
                    fake.indexed = True
                    code, out, err = fake.index
                    return ProcessResult(argv, code, out, err, 0.0)
                assert tool == "list_projects", argv
                return ProcessResult(argv, 0, fake.after if fake.indexed else fake.before, "", 0.0)

        monkeypatch.setattr(cbm, "ProcessRunner", Runner)
        exe = tmp_path / "codebase-memory-mcp.exe"
        exe.write_bytes(b"MZ")
        return str(exe)

    def tools(self):
        return [c[2] for c in self.calls if c[0] == "cli"]


# ------------------------------------------------------------ index_repository result parsing

def test_index_result_indexed_success():
    r = cbm.parse_index_result(0, json.dumps(INDEXED))
    assert r.ok and r.status == "indexed" and r.project == PROJECT and (r.nodes, r.edges) == (546, 1676)
    assert r.stats() == "546 nodes, 1676 edges" and not r.unrecognised


def test_index_result_payload_embedded_in_protocol_text():
    noisy = ("level=warn msg=mem.allocator.preloading_completed still_preloading=false\n"
             f"{envelope(json.dumps(INDEXED))}\ntrailing {{not json}} text\n")
    assert cbm.parse_index_result(0, noisy).ok
    bare = f"progress: 100% {{\n{json.dumps(INDEXED)}\ndone\n"
    r = cbm.parse_index_result(0, bare)
    assert r.ok and r.project == PROJECT
    progress = json.dumps({"status": "indexing", "done": 10}) + "\n" + json.dumps(INDEXED)
    assert cbm.parse_index_result(0, progress).status == "indexed", "the final result object wins"


def test_index_result_nonzero_exit_is_a_failure_even_with_an_indexed_payload():
    r = cbm.parse_index_result(1, json.dumps(INDEXED), "fatal: database is locked")
    assert not r.ok and not r.unrecognised and "exit 1" in r.detail and "database is locked" in r.detail
    assert not cbm.parse_index_result(None, json.dumps(INDEXED)).ok


@pytest.mark.parametrize("out", ["", "garbage", '{"status": "indexed"', json.dumps({"nodes": 5}), "[1, 2]"])
def test_index_result_malformed_response_is_not_success(out):
    r = cbm.parse_index_result(0, out)
    assert not r.ok and r.unrecognised and "recognisable" in r.detail


def test_index_result_error_status_and_error_envelope_are_failures():
    r = cbm.parse_index_result(0, json.dumps({"status": "error", "error": "repo_path does not exist"}))
    assert not r.ok and not r.unrecognised and "status=error" in r.detail and "does not exist" in r.detail
    r = cbm.parse_index_result(0, envelope(json.dumps(INDEXED), is_error=True))
    assert not r.ok and not r.unrecognised


# ------------------------------------------------------------ list_projects parsing

def test_list_projects_parses_real_text_json_and_envelope_forms():
    expected = {PROJECT: ROOT, OTHER[0]: OTHER[1]}
    assert cbm.parse_list_projects(LIST_TEXT) == expected, "unquoted roots (no spaces) must be parsed"
    assert cbm.parse_list_projects(list_json((PROJECT, ROOT), OTHER)) == expected
    assert cbm.parse_list_projects(envelope(LIST_TEXT)) == expected
    assert cbm.parse_list_projects(envelope(list_json((PROJECT, ROOT), OTHER))) == expected
    assert cbm.parse_list_projects(INDEX_STATUS_TEXT) == {}, "key/value lines are never taken as projects"
    assert cbm.parse_list_projects("") == {} and cbm.parse_list_projects("{broken") == {}
    assert cbm.project_for_root(cbm.parse_list_projects(LIST_TEXT), REPO) == PROJECT


def test_list_projects_requests_json_and_reads_every_page(monkeypatch, tmp_path):
    pages = {0: list_json(OTHER, has_more=True), 1: list_json((PROJECT, ROOT), offset=1)}
    seen = []

    class Runner:
        async def run(self, argv, *, timeout=None, **kw):
            args = json.loads(argv[4])
            seen.append(args)
            return ProcessResult(argv, 0, pages[args["offset"]], "", 0.0)
    monkeypatch.setattr(cbm, "ProcessRunner", Runner)
    projects = asyncio.run(cbm.list_projects("cbm.exe"))
    assert projects == {OTHER[0]: OTHER[1], PROJECT: ROOT}
    assert [(a["format"], a["offset"]) for a in seen] == [("json", 0), ("json", 1)] and seen[0]["limit"] == 500


def test_probe_reports_an_unquoted_root_as_indexed(monkeypatch, tmp_path):
    exe = FakeCLI(list_json((PROJECT, ROOT), OTHER)).install(monkeypatch, tmp_path)
    st = asyncio.run(cbm.probe(exe, REPO))
    assert st.available and st.repo_indexed is True and st.project == PROJECT
    exe = FakeCLI(LIST_TEXT).install(monkeypatch, tmp_path)
    assert asyncio.run(cbm.probe(exe, REPO)).project == PROJECT


# ------------------------------------------------------------ ensure_indexed

def test_ensure_indexed_already_indexed_repository_is_not_reindexed(monkeypatch, tmp_path):
    fake = FakeCLI(list_json((PROJECT, ROOT), OTHER))
    exe = fake.install(monkeypatch, tmp_path)
    assert asyncio.run(cbm.ensure_indexed(exe, REPO)) == (PROJECT, cbm.ALREADY_INDEXED)
    assert "index_repository" not in fake.tools()


def test_ensure_indexed_recognises_the_live_run_success(monkeypatch, tmp_path):
    """First V2 run: exit 0 + status=indexed, and list_projects then shows the root UNQUOTED."""
    payload = {k: v for k, v in INDEXED.items() if k != "project"}  # the id must come from list_projects
    fake = FakeCLI(LIST_TEXT_OTHER_ONLY, after=LIST_TEXT, index=(0, json.dumps(payload), ""))
    exe = fake.install(monkeypatch, tmp_path)
    name, detail = asyncio.run(cbm.ensure_indexed(exe, REPO))
    assert (name, detail) == (PROJECT, "546 nodes, 1676 edges")
    idx = next(c for c in fake.calls if c[2] == "index_repository")
    assert json.loads(idx[3]) == {"repo_path": ROOT}, "indexed by the exact absolute root"


def test_ensure_indexed_nonzero_exit_is_reported_as_failure(monkeypatch, tmp_path):
    fake = FakeCLI(list_json(OTHER), after=list_json((PROJECT, ROOT)), index=(2, "", "fatal: cannot open graph db"))
    exe = fake.install(monkeypatch, tmp_path)
    name, detail = asyncio.run(cbm.ensure_indexed(exe, REPO))
    assert name == "" and "exit 2" in detail and "cannot open graph db" in detail


def test_ensure_indexed_malformed_response(monkeypatch, tmp_path):
    fake = FakeCLI(list_json(OTHER), index=(0, "unexpected output", ""))
    exe = fake.install(monkeypatch, tmp_path)
    name, detail = asyncio.run(cbm.ensure_indexed(exe, REPO))
    assert name == "" and "recognisable" in detail
    # ...but exit 0 plus the exact root now being listed is real evidence of an index.
    fake = FakeCLI(list_json(OTHER), after=list_json((PROJECT, ROOT)), index=(0, "unexpected output", ""))
    exe = fake.install(monkeypatch, tmp_path)
    assert asyncio.run(cbm.ensure_indexed(exe, REPO)) == (PROJECT, "indexed now")


def test_ensure_indexed_payload_inside_protocol_text(monkeypatch, tmp_path):
    out = f"level=warn msg=preloading\n{envelope(json.dumps(INDEXED))}\n"
    fake = FakeCLI("", index=(0, out, "level=info msg=done"))  # listing unavailable: the result names the project
    exe = fake.install(monkeypatch, tmp_path)
    assert asyncio.run(cbm.ensure_indexed(exe, REPO)) == (PROJECT, "546 nodes, 1676 edges")


def test_ensure_indexed_never_takes_a_project_that_belongs_to_another_root(monkeypatch, tmp_path):
    fake = FakeCLI(list_json(OTHER), after=list_json(OTHER, (PROJECT, "C:/Users/Example/Elsewhere/SampleGame")))
    exe = fake.install(monkeypatch, tmp_path)
    name, detail = asyncio.run(cbm.ensure_indexed(exe, REPO))
    assert name == "" and "no project has the root" in detail
    wrong_root = dict(INDEXED, root_path="C:/Users/Example/Elsewhere/SampleGame")
    fake = FakeCLI("", index=(0, json.dumps(wrong_root), ""))
    exe = fake.install(monkeypatch, tmp_path)
    assert asyncio.run(cbm.ensure_indexed(exe, REPO))[0] == ""


# ------------------------------------------------------------ worker prompts

def _phase():
    return parse_phases([{"id": "P1", "title": "Core", "description": "obj",
                          "subtasks": [{"id": "S1", "title": "a"}, {"id": "S2", "title": "b"}],
                          "acceptance_criteria": ["works"]}])[0]


def _tools_block(prompt: str) -> str:
    return prompt.split("\nTOOLS:\n", 1)[1].split("\nRULES:\n", 1)[0]


def _how_block(prompt: str) -> str:
    return prompt.split("\nHOW TO WORK:\n", 1)[1].split("\nTOOLS:\n", 1)[0]


def _find(text: str, *concepts: tuple[str, ...]) -> int:
    """Index of the first instruction line mentioning every concept (a concept is a tuple of
    alternative words, case-insensitive), or -1. Checks meaning, not exact prose."""
    for i, line in enumerate(x.strip().lower() for x in text.splitlines() if x.strip()):
        if all(any(w.lower() in line for w in alts) for alts in concepts):
            return i
    return -1


def _v2_prompt(project=PROJECT, available=True):
    return build_phase_prompt(PhaseWorkSpec(phase=_phase(), repo_path=REPO, codebase_memory_available=available,
                                            codebase_memory_project=project))


def _named_tools(text: str) -> set[str]:
    params = {"root_path", "repo_path", "name_pattern", "qn_pattern", "file_pattern", "file_path", "function_name",
              "source_mode", "max_lines", "data_flow", "event_bus"}
    return set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text)) - params


GRAPH = ("graph", "codebase-memory")
MULTI_FILE = ("multiple source files", "several whole files", "broad")


def test_v2_prompt_makes_the_graph_the_primary_map_for_repo_wide_discovery():
    prompt = _v2_prompt()
    how, tools = _how_block(prompt), _tools_block(prompt)
    assert _find(how, GRAPH, ("primary",), ("map", "discovery")) >= 0, "the work procedure itself is graph-first"
    assert _find(tools, GRAPH, ("primary",)) >= 0
    for tool, question in [("get_architecture", ("subsystem", "layers", "entry points")), ("search_graph", ("symbol",)),
                           ("trace_path", ("calls", "callers")), ("get_file_outline", ("file",)),
                           ("search_code", ("pattern", "string")), ("query_graph", ("relationship",)),
                           ("get_code_snippet", ("symbol", "function"))]:
        assert _find(tools, (tool,), question) >= 0, f"{tool} is mapped to the question it answers"
    assert _named_tools(prompt) & TOOLS_0_11 >= {"get_architecture", "search_graph", "trace_path", "get_file_outline"}
    assert _named_tools(tools) <= TOOLS_0_11, f"not exposed by 0.11.0: {_named_tools(tools) - TOOLS_0_11}"


def test_v2_prompt_puts_graph_discovery_before_broad_multi_file_reads():
    prompt = _v2_prompt()
    how, tools = _how_block(prompt), _tools_block(prompt)
    assert _find(how, ("before",), MULTI_FILE, GRAPH) >= 0
    assert _find(how, ("then",), ("targeted", "only those")) >= 0
    assert _find(prompt, ("in full", "whole files"), ("avoid", "do not")) >= 0, "the whole-file anti-pattern is named"
    ask_first = _find(tools, ("first",), GRAPH)
    read_after = _find(tools, ("after",), ("targeted",), ("rg",))
    assert 0 <= ask_first < read_after, "graph questions come before targeted reads"
    assert _find(tools, ("get_code_snippet",), ("source_mode",), ("auto",)) >= 0
    assert _find(tools, ("source_mode full", "max_lines"), ("do not",), ("whole", "dump")) >= 0


def test_targeted_reads_rg_and_fallback_remain_allowed():
    tools = _tools_block(_v2_prompt())
    assert _find(tools, ("after",), ("targeted",), ("rg",)) >= 0
    assert _find(tools, ("direct reads",), ("already know", "exact file")) >= 0
    assert _find(tools, ("shell",), ("tests",), ("builds",)) >= 0
    assert _find(tools, ("performative", "just to satisfy")) >= 0, "graph calls must earn their keep"
    assert _find(tools, ("unavailable", "cannot answer"), ("rg", "source reads"), ("not block", "fall back")) >= 0
    assert _find(tools, ("never use rg", "do not use rg", "never read source")) == -1, "shell tools are not banned"


def test_resolved_project_identity_is_reused_not_rediscovered():
    tools = _tools_block(_v2_prompt())
    assert _find(tools, (PROJECT.lower(),), ("list_projects",), ("do not", "don't")) >= 0, \
        "a project verified by the orchestrator is used directly"
    assert _find(tools, (PROJECT.lower(),), ("index_status",), ("once",)) >= 0
    resolve = _find(tools, ("list_projects",), ("only when", "only if"))
    assert resolve >= 0, "project discovery is a conditional fallback"
    assert _find(tools, ("list_projects",), ("exactly this root",), ("index_repository",)) == resolve
    assert _find(tools, ("never guess the project name",)) >= 0
    assert _find(tools, ("whole session",), ("reuse", "remember"), ("do not repeat",), ("list_projects",)) >= 0
    assert _find(tools, ("another repository",)) >= 0
    unresolved = _tools_block(_v2_prompt(project=""))
    assert _find(unresolved, ("resolve",), ("once",)) >= 0 and PROJECT not in unresolved


def test_resumed_turns_keep_a_concise_graph_first_reminder():
    ph = _phase()
    for purpose in ("continue", "retry", "resume", "verify_fix"):
        for project in (PROJECT, ""):
            text = build_turn_prompt(purpose, ph, codebase_memory_available=True, codebase_memory_project=project)
            reminder = [x for x in text.splitlines() if "codebase-memory" in x]
            assert len(reminder) == 1 and len(reminder[0]) < 400, reminder
            r = reminder[0].lower()
            assert ("do not repeat" in r and "list_projects" in r) and (PROJECT.lower() in r) == bool(project)
            assert _find(r, ("first",), ("search_graph", "trace_path")) >= 0 and "targeted" in r
            assert _named_tools(reminder[0]) <= TOOLS_0_11
            assert "RULES:" not in text and "get_architecture" not in text, "the full tool guide is not repeated"
        plain = build_turn_prompt(purpose, ph)
        assert "codebase-memory" not in plain and "search_graph" not in plain and "list_projects" not in plain


def test_v2_prompt_without_codebase_memory_is_degraded_and_v1_prompt_is_unchanged():
    prompt = _v2_prompt(project="", available=False)
    assert CODEBASE_MEMORY_DEGRADED_LINE in prompt and "get_architecture" not in prompt and "PRIMARY" not in prompt
    assert _find(_how_block(prompt), GRAPH) == -1, "no graph-first procedure without codebase-memory"
    v1 = build_worker_prompt(WorkerTaskSpec(task=Task(id="T001", title="t"), repo_path=REPO,
                                            codebase_memory_available=True))
    assert "\n".join(CODEBASE_MEMORY_LINES) in v1
    assert _find(v1, ("primary map",)) == -1 and _find(v1, ("whole files", "in full")) == -1
    assert CODEBASE_MEMORY_LINES[0].startswith("- Prefer the codebase-memory MCP tools"), "V1 wording untouched"


# ------------------------------------------------------------ planner timeout default

def test_planner_timeout_defaults_to_240s_and_user_values_still_win():
    s = ClaudeSettings()
    assert s.plan_timeout_seconds == 240
    for purpose in ("PLAN", "REPLAN", "PLAN_V2", "REPLAN_V2"):
        assert limits_for(purpose, s).timeout == 240
    assert (s.review_timeout_seconds, s.checkpoint_timeout_seconds, s.smoke_timeout_seconds) == (60, 240, 45)
    assert OrchestrationSettings().verification_timeout_seconds == 900
    assert IntegrationSettings().codebase_memory_index_timeout_seconds == 900
    assert Settings.from_dict({"claude": {"plan_timeout_seconds": 120}}).claude.plan_timeout_seconds == 120
    assert Settings.from_dict({"claude": {"plan_timeout_seconds": 600}}).claude.plan_timeout_seconds == 600
