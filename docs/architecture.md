# Architecture and implementation notes

[Back to the project README](../README.md)

## Design notes

- **Threading:** the GUI runs on the Qt thread. The engine runs on its own thread with an
  asyncio (Proactor) loop, and events reach the GUI through a queued Qt signal
  (`gui/bridge.py`). Widgets never call CLIs.
- **State machine:** `orchestration/state_machine.py` holds the transition table.
  `orchestration/scheduler.py` has one handler per state, and the loop checks pause/stop
  between stages. Every transition is logged and saved in `state_transitions`.
- **Git attribution:** before and after every attempt the worktree is snapshotted into a tree
  object through a temporary `GIT_INDEX_FILE`. The attempt diff is `diff(pre_tree, post_tree)`,
  so pre-existing user changes are never credited to Qwen. The real index, working tree and
  refs are untouched; only git objects are written. The app never runs reset, clean, push or
  rebase. The working branch (`ai-orchestrator/run-<id>`, real worker only) is created with
  `git switch -c`, which carries uncommitted work along. Automatic commits are off and not
  implemented.
- **Context strategy:** each Qwen attempt is a fresh `exec`. A retry gets the task, the
  acceptance criteria, the previous attempt's summary and Claude's corrective instructions,
  never the old transcript. A PASSED task is reduced to a compact `TaskResult` that only its
  dependents receive.
- **Verification commands** from the plan run through `cmd.exe /c` on Windows. The command
  policy skips commands it flags as destructive and records them as requiring approval.
- **Metrics:** tok/s = Δ`llamacpp:tokens_predicted_total` / Δ`llamacpp:tokens_predicted_seconds_total`
  between polls. Claude cost is shown only as reported by the CLI (`total_cost_usd`).
- **Persistence:** SQLite at `%LOCALAPPDATA%\AIDevOrchestrator\orchestrator.db` (WAL mode).
  Settings are at `%APPDATA%\AIDevOrchestrator\settings.json`. No secrets are stored.

## Pipeline V2 (phase-based)

Choose it with the **Pipeline** selector next to PLAN, or under Settings → Orchestration. V1 stays the
default. Each run keeps the pipeline it was planned with, and V1 runs are never reinterpreted.

```
CLAUDE PLANS 3-6 PHASES
  PHASE n: ONE Qwen session  ->  turns (first + continuations while INCOMPLETE)
           -> local verify / fix turns in the same session  ->  verification
           -> CLAUDE CHECKPOINT: PASS | RETRY | BLOCKED | REPLAN      (last phase = FINAL AUDIT)
```

- **Phases:** each phase is one coherent objective with 3-8 internal subtasks. Qwen tracks them in its
  own todo list. Subtasks are not reviewed individually; Claude reviews only phase boundaries.
- **Session reuse:** the first turn of a phase is `exec`. Later turns use
  `exec resume <session> -`, which keeps Qwen's context. They cover continuations, local fixes and
  checkpoint RETRY feedback. `exec resume` has no `--cd`/`--sandbox`, so the process cwd is the
  repository and the sandbox is passed as `-c sandbox_mode="..."`. Sessions are persisted by Codex in the
  process-local `CODEX_HOME`, so no worker process has to stay alive between turns.
- **When a new session starts:** at every phase boundary; when a session cannot be resumed; when it
  reached *turns/session* (context pressure) or reported running out of context; or on the user's
  request (*Tools → V2: start a fresh Qwen session*). A new session mid-phase gets a compact handoff:
  last report, files changed so far, and subtasks done.
- **Checkpoint:** a bounded context like the V1 reviewer (see `orchestration/checkpoint.py`). It holds
  the phase's authored changes *since the phase started*, across all attempts. Earlier phases' and user
  changes are excluded, and generated/temporary paths (`node_modules/`, `dist/`, `build/`, `coverage/`,
  `scratch_smoke/`, caches, `*.map`, `*.bundle.js`, …) are only counted. The worker is told to put smoke
  artifacts in `scratch_smoke/` and delete them.
- **RETRY** enters **REPAIRING** (below). **REPLAN** closes the phase as SUPERSEDED (its work stays)
  and Claude revises only the future phases. PASSED phases are never changed. **BLOCKED** waits for the
  user; RESUME continues the phase's session with the user's guidance.
- **Subtask progress** comes only from Qwen's structured todo list (`S1: …` entries), persisted and shown
  the moment an entry is checked. Final-message prose/checklists never mark a subtask done; a COMPLETE
  report without the structured signals is logged as an inconsistency. A checkpoint PASS closes any
  unreported subtasks (marked `completed_via: checkpoint`).

### REPAIRING (bounded repair after a checkpoint RETRY)

```
REVIEWING --RETRY--> RETRY_PENDING -> REPAIRING -> VERIFYING -> REVIEWING
  REPAIRING: fresh Qwen repair session -> [turn -> protected tests -> full verification -> score] ...
             guard tripped -> Claude DIAGNOSE_BLOCKER -> REPAIR: new session with guidance | BLOCKED: halt
```

- **Fresh session:** a repair never extends the (large) implementation session. The new session gets a
  concise reconstruction: objective, acceptance criteria, completed subtasks (not to be redone),
  checkpoint findings, current and best-known failing assertions, files changed, protected tests.
  Within one repair session, rounds resume the same session with a short "round N: improved/unchanged/
  worse" prompt.
- **Evidence, not prose:** after every turn the phase verification runs and is **fingerprinted**: the
  failing assertion lines (`FAIL…`, `FAILED…`, `not ok`, `✗`, `●`, `error TS…`) are normalized (ANSI
  colors, timestamps, absolute/temp paths, `file:line`, addresses, decimals and parenthesized
  measurements removed; thresholds like `20 s` kept) and hashed. A **score** orders build failure >
  timeout/crash > number of failing assertions. "Found the root cause" changes nothing; only a changed
  or better verification result resets the stuck counter.
- **Best-known state:** a strictly better score is recorded as a snapshot tree (the existing
  temporary-index snapshot; real index, HEAD and refs untouched). A result worse than it gets
  `repair_regression_grace_rounds` round(s) to recover, then the snapshot is restored: only authored
  files that differ between the two snapshots are rewritten/removed through a temporary index, so
  pre-existing user changes (part of both snapshots), ignored and generated paths are never touched.
  Changed/untracked files of a best-known state are also kept as exact `--no-filters` blobs, so a
  worker-created file that was later changed or deleted comes back byte for byte (the tree alone holds
  EOL-normalized content under `core.autocrlf`).
  The regressed state is kept as a logged snapshot tree (`git cat-file -p <tree>:<path>` recovers it).
- **Protected acceptance tests:** files named by the phase's verification commands (and test files under
  a named directory), plus existing test files the phase created/changed/planned, are stored as exact
  blobs when the repair starts. Edits to them are reverted after every turn and reported to the worker,
  unless the checkpoint (`authorized_test_changes`) or a diagnosis authorized them. New test files are
  allowed.
- **Guard** (Settings → Orchestration → V2 repair limits; any limit stops the repair session):
  `repair_max_wall_seconds` 1200, `repair_max_tool_calls` 30, `repair_max_same_failure_rounds` 3,
  `repair_max_no_improvement_rounds` 4 (rounds whose score is not strictly better than the best-known
  state - a different fingerprint, or beating only the previous round, is not progress),
  `repair_regression_grace_rounds` 1, `repair_max_new_probe_files` 3 (new scratch/probe/debug files
  without progress; counted, never deleted), a second best-known restoration, or a worker BLOCKED report.
- **DIAGNOSE_BLOCKER:** a cheap Claude call (no tools, `diagnose_*` limits in the Claude settings) with a
  byte-bounded packet (phase goal, checkpoint feedback, completed subtasks, current/best failing
  assertions, changed files, bounded repair diff, recent rounds, guard reason). It returns a strict JSON
  diagnosis (`diagnosis`, `recommended_actions`, `avoid`, `target_files`, `target_tests`, `status`).
  REPAIR starts a new session with that guidance and a fresh budget (at most `repair_max_diagnoses` = 2
  per repair attempt, then the best-known state goes to the checkpoint); BLOCKED halts the run until
  the user resumes with guidance.
- **Recovery:** the repair state (`pipeline_state.repair`) persists attempt, counters incl. the tool
  calls/time of a turn in flight, fingerprint, best-known evidence and snapshot id, protected blobs,
  diagnoses and pending escalation. After a restart the repair continues (the repair session is resumed
  with a "resuming the repair" prompt); the phase and completed subtasks are never restarted.
- **codebase-memory:** before the first phase the repository is looked up by its exact absolute root. If
  it is missing, it is indexed with `index_repository` (success = exit 0 and a result with
  `status: indexed`; the exact project id is then resolved by root). The worker gets the exact project
  name and uses it directly (one `index_status`; `list_projects` is only a fallback) and never guesses it
  from the folder name. It keeps that name for the whole phase session. The graph is its primary map:
  `get_architecture` / `search_graph` / `trace_path` / `get_file_outline` narrow things down first, then
  it reads only the targeted functions or line ranges (`get_code_snippet`, `rg`, direct reads). Resumed
  turns repeat this as a one-line reminder, and it falls back to `rg` when the graph can't help.
- **Recovery:** the Qwen session id is saved as soon as Codex reports it. After STOP, a crash or a
  killed worker, RESUME continues the phase in the same session. Interrupted attempts don't count
  against the retry limit. PAUSE takes effect between worker turns.
- **Persistence:** migration 2 only *adds* `runs.pipeline_version` (default `'v1'`) and
  `runs.pipeline_state_json`. Phases, checkpoints and turns use the existing tables.

### V4 pass 1: verification contracts and human verification

```
REVIEWING --ASK_USER--> AWAITING_HUMAN --(user answers)--> Claude HUMAN_FOLLOWUP
    PASS -> ADVANCING    RETRY -> RETRY_PENDING -> REPAIRING (the existing bounded repair)    BLOCKED -> BLOCKED
```

- **Verification contract** (optional, per phase, written by Claude when planning): *what* evidence would
  convincingly show the phase succeeded, never *how* a tool obtains it. `success_dimensions`, semantic
  `evidence_requirements` (`{id, goal, kind, required, reason}`, e.g. `runtime_behavior`, `visual_evidence`)
  and a `human_gate` (`{allowed, reason, checklist}`). The checkpoint judges each requirement
  (`evidence_assessment`); the worker sees the evidence goals. Phases without a contract behave exactly as
  before. Evidence execution was added in pass 2.
- **ASK_USER** is a checkpoint verdict, offered only when *Settings → Orchestration → Allow Claude to request
  human verification* is on **and** the phase's contract opens its human gate. Claude uses it only when a
  person's observation materially decides acceptance and the automated evidence cannot. Its questions are
  generated for that phase. Otherwise Claude decides PASS / RETRY / BLOCKED / REPLAN from the evidence.
- **AWAITING_HUMAN** is a waiting state (neither executing nor BLOCKED). The run parks there with the exact
  request (reason, questions, observations, checklist, automated evidence summary) persisted in
  `pipeline_state.human`. A restart leaves it waiting and starts nothing; STOP/RESUME returns to it.
- **Answer:** the panel above the details shows the request; *Looks good* / *Issue found* (a note is
  required) / *Cannot verify*, plus a free-text note that is stored as written and never parsed.
- **HUMAN_FOLLOWUP:** a bounded Claude call (no tools, `human_followup_*` limits) with the phase, contract,
  automated evidence, the ASK_USER checkpoint, the request, the answer, the repository summary and prior
  diagnoses. It returns PASS / RETRY / BLOCKED only (no ASK_USER recursion). Its result replaces the attempt's
  ASK_USER review (both are in the `reviews` table), so RETRY enters the existing repair: fresh session,
  completed subtasks kept, the phase is not reset, and the repair prompt carries the user's note,
  `do_not_redo` and the derived criteria. If the call is interrupted, RESUME re-runs it for the recorded answer.
- **Derived human criteria** (`HF-<phase>-NN`) are stored on the phase, separate from its acceptance
  criteria, and included in every later repair, diagnosis and checkpoint prompt of that phase.
- **Issue metadata:** checkpoint findings may carry `severity` (critical, functional, regression, polish,
  subjective, micro_deviation) and `disposition` (repair, ask_user, accept, blocked). It is descriptive only;
  the orchestrator never derives a decision from it.
- **Compatibility:** no schema migration. Everything V4 is additive JSON with defaults, so V1 runs and V2/V3
  runs and databases load unchanged.

### V4 pass 2: adaptive evidence discovery and execution

```
VERIFYING -> COLLECTING_EVIDENCE -> REVIEWING           (only if the contract lists evidence requirements)
  COLLECTING_EVIDENCE: capability discovery -> Claude EVIDENCE_PLAN -> bounded execution -> evidence bundle
```

- **Capabilities** (`app/evidence/model.py`) are things AI Studio can really do here. Each has a semantic,
  open-vocabulary `kind` and an honest confidence: **CONFIRMED** (it actually ran in this run),
  **INFERRED** (declared, program found, not proven) or **UNAVAILABLE** (explicitly checked; the reason is
  recorded).
- **Providers** (`app/evidence/providers/`) implement `discover` / `validate` / `execute`, plus an optional
  lifecycle for things that must run while evidence is observed (`start → ready → capture → stop`).
  The registry (`app/evidence/registry.py`) runs a bounded discovery across them. The scheduler only knows
  capabilities, plans and results, so a new provider plugs in without touching it. Pass 2 ships two providers:
  - **command:** the phase's and earlier phases' verification commands; commands configured under Settings
    → *V4 evidence commands*; commands and **services** declared in the repository's
    `.aistudio/evidence.json`; and `package.json` scripts. Destructive commands, scripts with persistent side
    effects and long-running scripts are UNAVAILABLE. Arguments are accepted only where declared, and each must
    match the declared pattern.
  - **repository_files:** read-only `exists` / `contains` checks inside the repository.
- **Discovery** runs for every phase attempt (pass 3; capabilities can change with the work). It is time- and
  size-bounded, reads metadata only, never runs the project, and never decides what evidence is needed.
- **EVIDENCE_PLAN** is a bounded Claude call with no tools. Claude plans only with the capability ids that
  discovery listed. The orchestrator validates the plan:
  - unknown or unavailable ids, free-form commands and undeclared arguments are rejected;
  - `required` always comes from the contract, and a requirement the planner skipped stays as UNAVAILABLE;
  - total steps are bounded;
  - a build, static check or file check can only give *indirect* coverage of a runtime, visual or
    interaction requirement.

  The plan is reused while the contract, the derived criteria and the capabilities are unchanged. With no
  usable capability at all, no Claude call is made.
- **Execution** (`app/evidence/executor.py`) is verification only. It is bounded by wall time, per-step
  timeout, step count, output bytes, artifacts and retries (Settings → Orchestration → *V4 evidence
  limits*). Each requirement ends SATISFIED, FAILED, UNAVAILABLE, INCONCLUSIVE or SKIPPED. Only a direct,
  executed pass is SATISFIED. Every step is persisted first as RUNNING, so after a crash an interrupted step
  is re-run and never trusted, and completed steps are not repeated. Every service started is stopped, and
  leftovers of a crashed collection are killed only when pid **and** creation time match. If a step changes
  authored repository files, they are restored and that evidence becomes INCONCLUSIVE.
- **Artifacts** (full step and service output) are files under AI Studio's data folder, never in the target
  repository. The run state holds only their metadata and short output tails.
- **Checkpoint:** the prompt lists every requirement's result right after the contract, required gaps
  first. Capabilities, the plan and artifact references are compacted. Claude answers `evidence_sufficient`
  and `missing_evidence`. Green builds or tests are explicitly not proof of runtime, visual or interaction
  properties. A PASS with unverified required evidence records `UNVERIFIED …` in the phase's remaining risks.
- **Human gate:** an ASK_USER request carries the evidence lines (✓ / ✗ / ? / ~) and what could not be
  established. HUMAN_FOLLOWUP receives the evidence results. A human RETRY still enters the existing repair.

### V4 pass 3: evidence hardening, first real providers, implementation guard

```
REVIEWING -> CONSISTENCY_REVIEW -> ADVANCING | RETRY_PENDING | AWAITING_HUMAN | BLOCKED
             (only when a PASS contradicts its own evidence judgment; never recursive)
WORKER_RUNNING: implementation guard -> MID_PHASE_DIAGNOSE -> fresh worker session  (or BLOCKED)
```

- **Consistency review** (`app/orchestration/consistency.py`): a checkpoint that answers PASS with
  `evidence_sufficient=false` while required evidence is unresolved (or PASS over a FAILED required result)
  gets one bounded, tool-free Claude CONSISTENCY_REVIEW with a compact packet. It answers PASS / RETRY /
  BLOCKED (ASK_USER when the human gate is open). A PASS must justify every unresolved requirement
  (`not_needed` / `substitute_evidence` + reason) or it is rejected; accepted requirements stay recorded as
  UNVERIFIED. The adjudication is persisted (`pipeline_state.consistency`); a restart repeats only the review.
  Limits: Settings → Claude → *V4 consistency review* (timeout, turns, budget, prompt bytes).
- **Human-derived criteria are evidence requirements:** `HF-<phase>-NN` criteria (with their evidence kind,
  the request/feedback they came from, a status and a bounded history) are planned, executed and tracked
  like contract requirements on later attempts. The original acceptance criteria are never modified. The
  same issue reported again is not duplicated (deterministic normalization; word order, negation and every
  content word are kept).
- **Freshness** (`app/evidence/freshness.py`): every executed step records what it ran against (repository
  fingerprint - a git snapshot tree or a bounded manifest - the capability definition, file inputs and a
  signature). Evidence whose inputs changed is STALE: it becomes INCONCLUSIVE (its old status is kept in
  `stale_status`) and never satisfies a phase. It is re-checked before the checkpoint, the consistency
  review, the human follow-up and PASS. A step of an earlier attempt is reused only when its inputs are
  provably unchanged (a file check stays valid when an unrelated file changes; a runtime check does not).
- **Bounded replanning:** when an evidence *strategy* cannot be executed (a capability turns out unavailable,
  a service never starts or never becomes ready, a provider error, or a capability of the needed kind got
  confirmed later), only those requirements are re-planned, at most `evidence_max_replans` times per attempt
  (default 1). The failed capability is marked unavailable first, and the count is persisted. An ordinary
  FAILED check is evidence of a defect and is never replanned. A check that ran cleanly but never showed the
  output the plan expected (exit 0, marker missing - cause `unobserved`, e.g. a command that does not print
  its test results) may be re-planned the same way; unless the replan finds steps to execute, it stays FAILED.
- **Readiness probes** (`app/evidence/readiness.py`): `process_alive`, `stdout_contains`, `tcp_port`,
  `http_status`, `file_exists` - declarative, validated (localhost only, no credentials, no redirects, no
  proxy, bounded timeouts, repository-relative paths), never a shell command.
- **Artifact retention:** artifacts stay under `<app data>/evidence`, every path is checked to stay inside it,
  binary artifacts (images) are never truncated, and a per-run byte budget applies. Cleanup (at startup and
  when a run completes) deletes by age, per-run bytes and total bytes - never artifacts of runs that can still
  continue, never anything outside that folder. Settings → Orchestration → *Evidence artifacts*.
- **Non-git projects:** a bounded manifest guard detects changes made by evidence steps; within the byte budget
  it keeps an exact snapshot and restores; beyond it, it reports that restoration cannot be guaranteed.
- **Providers** (all behind the registry; the scheduler has no technology-specific branch):
  - **runtime** - applications the project declares under `runtimes` in `.aistudio/evidence.json`: launch with
    a readiness probe, re-check readiness, "still running", output + log capture, local HTTP endpoints and
    declared actions while it runs. Observing capabilities declare `requires_resource`.
  - **visual** - `visual_capture` and `image_analysis` are separate. On Windows a real capture backend
    records a window of a resource AI Studio itself started (`app/evidence/wincapture.py`; never the desktop or
    another application). A capture alone is never visual evidence; no image analysis backend exists yet, so
    analysis is UNAVAILABLE.
  - **unreal** - only when a `.uproject` exists: descriptor check, UnrealBuildTool build (the header tool runs
    as part of it; failures are reduced to root diagnostics - UHT first - with the full log as an artifact),
    automation tests, declared read-only commandlets (asset-rewriting ones are refused), game/editor launch with
    log capture and crash detection. Automation runs headless with `-TestExit="Automation Test Queue Empty"
    -stdout -FullStdOutLogOutput` and a report directory created for that run; it passes only when at least one
    test under the filter (whole dotted segments; `A+B` = either) completed and every one of them reported
    success - never on the exit code alone, never on an unrelated test. Results come from stdout, the report, or
    the project log only as far as this run provably wrote it (`app/evidence/freshlog.py`: snapshot before the
    launch, then only a rotated/new log or the appended part; a stale `Result={Success}` never counts, and
    nothing in the project is deleted). Engine: provider setting `unreal.engine_root`, the Epic
    launcher/registry, or an enclosing engine tree. No Blueprint authoring.
  - **browser** (optional, off unless `browser.enabled=true`) - declared local pages only, bounded action
    lists, no password entry; it needs an existing Playwright + browser installation (never installed by AI
    Studio) and is UNAVAILABLE otherwise.
- **Provenance:** each capability records who discovered it, the provider version, where it was declared (and
  whether the worker changed that file in this phase), the executable, when it was last validated and which
  steps used it.
- **Only harness observations count:** a worker's report, and logs its own runs left behind, are never evidence.
  A file check of a generated file (build output, logs, caches, anything the repository ignores) counts only if
  an evidence step of the current collection wrote it. A verification helper script the worker writes is used
  only when declared in `.aistudio/evidence.json`; AI Studio then runs it and reads its output itself, and the
  script's content hash is part of the capability definition, so a changed script makes its earlier results
  stale. A requirement result is labelled *reused* only when none of its steps ran now.
- **Implementation guard:** a normal implementation session is bounded by wall time (2400 s), tool calls (50)
  and time without planned-subtask progress (1200 s; only a structured todo completion of a PLANNED subtask
  counts). A trip stops the turn; Claude MID_PHASE_DIAGNOSE (no tools, short packet) guides a fresh session.
  After `implementation_max_diagnoses` (2) another trip blocks the phase. When every planned subtask is done
  and the phase verification is green, an INCOMPLETE report or non-zero exit no longer keeps the worker going.
- **Human questions** are short, deduplicated and bounded (`human_max_questions`); a question Claude ties to a
  requirement automation already proved is not asked. The request shows *PLEASE CHECK*, *AUTOMATION COULD NOT
  ESTABLISH* and *AUTOMATION ESTABLISHED*.

## V5: worker profiles

The scheduler talks to a `WorkerProvider` only. *Which* Codex build, `CODEX_HOME` and child environment a
run's worker uses is a **worker profile** (`app/workers/profiles.py`); provider and model live in that profile's
own `CODEX_HOME/config.toml`, which stays authoritative. No code branches on a provider.

| Concept | Where | Notes |
|---|---|---|
| Worker profile | `Settings.worker_profiles` (`settings.json`) | id, name, runtime (`codex`), executable, `CODEX_HOME`, non-secret extra env, required env **names**, optional `--model`, display metadata |
| Legacy profile `legacy-ninfer-qwen` | `Settings.worker` (the Worker tab) | always present, derived from the V4 settings; an installation without profiles behaves exactly like V4 |
| Run snapshot | `runs.worker_snapshot_json` | locked (write-once, committed) before a run's first worker process; every resume uses it |
| Launch spec / launcher | `app/workers/launcher.py` | snapshot + global policy (sandbox, timeout) → runtime adapter (`codex` → `CodexNInferWorker`) |

- **Resume safety.** A run locks the *default* profile at its first START. Changing the default, editing or
  re-verifying a profile only affects NEW runs. A V4 run (started before V5, no snapshot) is bound to the legacy
  profile, never to the current default. If the locked profile cannot be used (deleted, disabled, executable or
  `CODEX_HOME` missing, a required environment variable unset, verification revoked by a failing test) the run
  stays halted with `WORKER_PROFILE_UNAVAILABLE: …` — nothing ever falls back to another profile.
- **External vs managed.** *Register Existing* references an existing `CODEX_HOME` (e.g. `~\.codex-lmstudio`);
  AI Studio never rewrites its `config.toml`. *Add* / *Duplicate (copy home)* creates a NEW managed home under
  `~\.codex-profiles\<id>` by copying a template's `config.toml` (+ `AGENTS.md`; never sessions or logs) or by
  writing a minimal one. An existing folder is never overwritten and a profile folder is never deleted.
- **Secrets.** Extra env entries that look like credentials are refused for new profiles; list the variable NAME
  under *required env* (e.g. `OPENROUTER_API_KEY`, auto-detected from a `config.toml` auth command) and keep the
  value in your user environment. Snapshots, `worker_calls` rows and logs never contain secret values.
- **Validation.** *Test (quick)* checks the executable (never the normal `codex` launcher / npm shim / scripts),
  `CODEX_HOME`, `config.toml` parsing, required env names and shared homes, without launching anything. *Full
  test* is the old worker smoke test run through the profile's own launch path; its result (bound to that
  executable + `CODEX_HOME`) gates REAL runs of that profile, and a failure revokes it.
- **Executables.** The legacy profile keeps the V4 rule (`*ninfer*.exe`). Other profiles accept any dedicated
  Codex build (e.g. `codex-lmstudio.exe`) with the same launcher/npm/script protections.
- **History.** `worker_calls.worker_profile_id/_revision` record the profile of every worker call.
- **Prepared, not implemented:** per-phase profile overrides (`RunWorkerProfiles.phase_overrides`) and an explicit
  "retry with another worker profile" action. Nothing switches profiles automatically.

## V5.1: worker runtimes (Codex + Cline)

A worker profile now has a **runtime**. `app/workers/runtimes.py` is the one registry that maps it to a
`WorkerRuntime` (`codex` → `CodexWorkerRuntime`, `cline` → `ClineWorkerRuntime`): profile checks, resume-time
checks, the explicit CLI probe and the worker adapter. The schedulers only see the `WorkerProvider` contract
(`run_turn` / `execute_task` / `session_resume_supported` / `cancel` + callbacks), pinned by
`tests/test_worker_runtime_contract.py`. Profiles saved before V5.1 are Codex profiles; the Codex launch is
byte-identical (characterization tests). No database migration: snapshots already carried `runtime`.

**Cline CLI behavior observed during development (3.0.65)** — native `…\npm\node_modules\cline\node_modules\@cline\
cli-windows-x64\bin\cline.exe` (the npm `cline.cmd` shim is refused like every batch file):

| Concern | Behaviour | How AI Studio uses it |
|---|---|---|
| Start | `cline -y --json -c <repo> -P <provider> -m <model> [--config <dir>] [args] "<header>"` | `--json` requires a prompt **argument**; piped stdin is appended to it, so the full prompt goes over stdin (no 32K command-line limit, not in the recorded argv) |
| `-y` (yolo) | documented in Cline's README, hidden from `--help` | **required**: forces the local backend — the agent runs inside AI Studio's child. Without it a one-shot run attaches to the shared hub daemon and runs *inside the user's own Cline process*. Tools: read_files, run_commands, editor, submit_and_exit (no sub-agents/teams, no ask_question) |
| Output | NDJSON `{"ts","type"}`: `agent_event` (content_start/end for text, reasoning, tool; usage; done; error; notice), `hook_event`, `run_result`, `run_aborted`; `{"type":"error"}` on stderr | normalized to `WorkerEvent` (`app/workers/events.py`), the same vocabulary as Codex; streamed chunks are not logged, raw events are redacted |
| Session id | not in the stream; `cline history --json` lists sessions with the **pid** of the process that ran them | looked up by the exact pid of the worker process (local backend ⇒ our pid), persisted at once via `on_session` |
| Resume | `--id <session>` forces the interactive TUI (the prompt is dropped); `--acp` can load sessions but authenticates only Cline-account OAuth providers and may open a browser login | **not supported headlessly** → `session_resume_supported()` is False; the pipeline's existing, logged path starts a new Cline session with the compact handoff for every turn. The run itself (runtime, profile, provider, model) always resumes from its snapshot |
| Exit codes | 0 only for `finishReason == "completed"`; 1 otherwise (incl. its own timeout) | failure text includes the finish reason |
| Cancel / timeout | — | ProcessRunner: graceful, then `taskkill /PID <worker> /T /F` — only the worker's tree, never an image-name kill |
| Credentials | Cline's own configuration (`cline auth`) | the profile selects `-P` / `-m`; AI Studio reads only provider ids and saved model names (for the form) |

**Capabilities (never faked).** Cline reports exact tool calls, but no structured todo list and has no command
sandbox: the implementation guard keeps its wall-clock and tool-call limits and skips the no-subtask-progress
limit (`WorkerCapabilities`, logged when a run starts); START warns that the worker is not sandboxed. Codex keeps
every limit exactly as before.

**Profiles / UI.** Settings > Worker Profiles: *Add…* starts with **Worker runtime**; a Cline profile has the native
executable, provider, model (prefilled from Cline's saved model), optional `--config` directory, extra env,
required env names, extra CLI args and notes — no CODEX_HOME. The list shows `Runtime: Codex / Cline` per profile.
*Test (quick)* also checks that the Cline CLI responds (`--version`); *Full test…* (one tiny prompt; warns that it
may use paid credits) runs only when asked. A profile's runtime is fixed at creation.

**No fallback.** A run locked to a Cline profile is never continued with Codex (or another profile); an unusable
Cline snapshot (executable/config dir missing, disabled, deleted, revoked verification, runtime changed) stops with
`WORKER_PROFILE_UNAVAILABLE`.
