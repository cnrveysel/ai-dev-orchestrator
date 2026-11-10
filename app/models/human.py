"""Pipeline V4 human gate: Claude asks the user to verify what only a person can judge.

  request           ASK_USER from a checkpoint: reason, questions generated for THIS phase,
                    observations, checklist, plus a compact summary of the automated evidence
                    at that moment (so a restart shows exactly what was asked, and why)
  feedback          the user's structured answer: LOOKS_GOOD | ISSUE_FOUND | CANNOT_VERIFY and a
                    free-text note that is stored as given and never parsed
  follow-up         Claude HUMAN_FOLLOWUP adjudicates the answer: PASS | RETRY | BLOCKED only
  derived criteria  phase-scoped acceptance criteria Claude derives from the feedback
                    (HF-<phase>-NN), kept separate from the phase's original criteria. V4 pass 3:
                    they are first-class evidence requirements of the phase's later attempts, with
                    provenance (the request/feedback they came from), an evidence kind, a status
                    (open | verified | failed | unverified) and a bounded history

The gate state lives in run.pipeline_state["human"][<phase id>]; derived criteria live on the
phase. Both are plain JSON, so older runs simply have none.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.evidence.model import FAILED, is_satisfied, result_line

VERDICTS = ("LOOKS_GOOD", "ISSUE_FOUND", "CANNOT_VERIFY")
AWAITING = "awaiting"  # the request is shown; no answer yet
FEEDBACK_RECEIVED = "feedback_received"  # answered; HUMAN_FOLLOWUP not completed yet
RESOLVED = "resolved"  # HUMAN_FOLLOWUP decided
SUPERSEDED = "superseded"  # a newer checkpoint replaced an unanswered request

MAX_QUESTIONS = 8
MAX_OBSERVATIONS = 8
MAX_CHECKLIST = 12
MAX_EVIDENCE_LINES = 16
MAX_HISTORY = 10
MAX_NOTE = 4000
MAX_CRITERIA = 20


def _s(v: Any, limit: int = 1000) -> str:
    return str(v).strip()[:limit] if v is not None else ""


def _strs(v: Any, n: int, limit: int = 400) -> list[str]:
    out: list[str] = []
    for x in v if isinstance(v, list) else []:
        t = _s(x, limit)
        if t and t not in out:
            out.append(t)
    return out[:n]


def short(text: str, n: int = 160) -> str:
    """One-line, bounded rendering for logs (never the whole payload)."""
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[:n - 3] + "..."


@dataclass
class HumanRequest:
    phase_id: str
    reason: str
    questions: list[str]
    observations: list[str] = field(default_factory=list)
    checklist: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)  # automated evidence summary when Claude asked
    phase_title: str = ""
    attempt: int = 0  # the reviewed phase attempt whose checkpoint asked
    checkpoint_summary: str = ""
    final: bool = False
    created_at: float = 0.0
    missing: list[str] = field(default_factory=list)  # V4 pass 2: what the automated evidence could not establish
    # --- V4 pass 3 ---
    established: list[str] = field(default_factory=list)  # AUTOMATION ESTABLISHED (fresh SATISFIED results)
    targets: list[str] = field(default_factory=list)  # requirement id per question ("" = not tied to one)
    dropped: list[str] = field(default_factory=list)  # questions not asked (duplicate / proven / over the limit)

    @property
    def id(self) -> str:
        return f"{self.phase_id}-a{self.attempt}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "HumanRequest | None":
        if not isinstance(d, dict) or not d.get("phase_id"):
            return None
        questions = _strs(d.get("questions"), MAX_QUESTIONS)
        targets = [str(t or "")[:40] for t in (d.get("targets") or [])][:len(questions)]
        return cls(phase_id=_s(d["phase_id"], 40), reason=_s(d.get("reason")),
                   questions=questions, observations=_strs(d.get("observations"), MAX_OBSERVATIONS),
                   checklist=_strs(d.get("checklist"), MAX_CHECKLIST),
                   evidence=_strs(d.get("evidence"), MAX_EVIDENCE_LINES, 300), phase_title=_s(d.get("phase_title"), 300),
                   attempt=int(d.get("attempt") or 0), checkpoint_summary=_s(d.get("checkpoint_summary"), 2000),
                   final=bool(d.get("final")), created_at=float(d.get("created_at") or 0.0),
                   missing=_strs(d.get("missing"), MAX_OBSERVATIONS),
                   established=_strs(d.get("established"), MAX_OBSERVATIONS, 300),
                   targets=targets, dropped=_strs(d.get("dropped"), MAX_QUESTIONS, 300))


@dataclass
class HumanFeedback:
    phase_id: str
    verdict: str  # LOOKS_GOOD | ISSUE_FOUND | CANNOT_VERIFY
    note: str = ""
    questions: list[str] = field(default_factory=list)  # exactly the questions the user answered
    created_at: float = 0.0
    request_id: str = ""

    @classmethod
    def create(cls, request: HumanRequest, verdict: str, note: str = "") -> "HumanFeedback":
        v = str(verdict or "").strip().upper()
        if v not in VERDICTS:
            raise ValueError(f"invalid human verdict {verdict!r} (expected {', '.join(VERDICTS)})")
        text = str(note or "").strip()
        if v == "ISSUE_FOUND" and not text:
            raise ValueError("describe what you observed: a note is required with 'Issue found'")
        return cls(request.phase_id, v, text[:MAX_NOTE], list(request.questions), time.time(), request.id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "HumanFeedback | None":
        if not isinstance(d, dict) or str(d.get("verdict") or "").upper() not in VERDICTS:
            return None
        return cls(phase_id=_s(d.get("phase_id"), 40), verdict=str(d["verdict"]).upper(),
                   note=_s(d.get("note"), MAX_NOTE), questions=_strs(d.get("questions"), MAX_QUESTIONS),
                   created_at=float(d.get("created_at") or 0.0), request_id=_s(d.get("request_id"), 60))


@dataclass
class HumanGateState:
    """The human gate of ONE phase: the current round plus a bounded history of earlier rounds."""
    phase_id: str
    status: str = AWAITING
    request: HumanRequest | None = None
    feedback: HumanFeedback | None = None
    followup: dict[str, Any] = field(default_factory=dict)  # decision, summary, derived criterion ids, at
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pending(self) -> bool:
        return self.request is not None and self.status in (AWAITING, FEEDBACK_RECEIVED)

    def _round(self, status: str) -> dict[str, Any]:
        return {"status": status, "request": self.request.to_dict() if self.request else None,
                "feedback": self.feedback.to_dict() if self.feedback else None, "followup": dict(self.followup)}

    def _archive(self, status: str) -> None:
        if self.request is not None:
            self.history = (self.history + [self._round(status)])[-MAX_HISTORY:]
        self.request, self.feedback, self.followup = None, None, {}

    def begin(self, request: HumanRequest) -> None:
        if self.request is not None:
            self._archive(RESOLVED if self.status == RESOLVED else SUPERSEDED)
        self.request, self.status = request, AWAITING

    def supersede_pending(self) -> bool:
        """A new checkpoint of the phase replaces a request that was never adjudicated."""
        if not self.pending:
            return False
        self._archive(SUPERSEDED)
        self.status = SUPERSEDED
        return True

    def record_feedback(self, feedback: HumanFeedback) -> None:
        self.feedback, self.status = feedback, FEEDBACK_RECEIVED

    def resolve(self, decision: str, summary: str, criteria_ids: list[str]) -> None:
        self.followup = {"decision": decision, "summary": _s(summary, 2000), "derived_criteria": list(criteria_ids),
                         "at": time.time()}
        self.status = RESOLVED

    def rounds(self) -> list[dict[str, Any]]:
        """Every round of this phase, oldest first (the current one last)."""
        cur = [self._round(self.status)] if self.request is not None else []
        return list(self.history) + cur

    def to_dict(self) -> dict[str, Any]:
        return {"phase_id": self.phase_id, "status": self.status,
                "request": self.request.to_dict() if self.request else None,
                "feedback": self.feedback.to_dict() if self.feedback else None,
                "followup": dict(self.followup), "history": list(self.history)}

    @classmethod
    def from_dict(cls, d: Any) -> "HumanGateState | None":
        if not isinstance(d, dict) or not d.get("phase_id"):
            return None
        return cls(phase_id=_s(d["phase_id"], 40), status=_s(d.get("status"), 40) or AWAITING,
                   request=HumanRequest.from_dict(d.get("request")), feedback=HumanFeedback.from_dict(d.get("feedback")),
                   followup=dict(d.get("followup") or {}),
                   history=[x for x in d.get("history") or [] if isinstance(x, dict)][-MAX_HISTORY:])


CRITERION_STATUSES = ("open", "verified", "failed", "unverified")
MAX_CRITERION_HISTORY = 10


@dataclass
class HumanCriterion:
    """An acceptance criterion derived from human feedback; binding for the rest of the phase and, from
    V4 pass 3 on, an evidence requirement of the phase's later attempts (id HF-<phase>-NN)."""
    id: str
    text: str
    verdict: str = ""  # the feedback verdict it was derived from
    note: str = ""  # the user's note (bounded)
    attempt: int = 0
    created_at: float = 0.0
    # --- V4 pass 3 (additive; older criteria get the defaults) ---
    kind: str = ""  # semantic evidence kind for planning ("" = unspecified)
    rationale: str = ""
    source: str = "human_followup"
    request_id: str = ""  # the ASK_USER request whose answer revealed it (e.g. P2-a1)
    feedback_ref: str = ""  # <request id>@<feedback created_at>: the originating feedback
    feedback_refs: list[str] = field(default_factory=list)  # every feedback that reported it (deduplicated reports)
    reports: int = 1
    status: str = "open"  # open | verified | failed | unverified - from its latest evidence result
    evidence: dict[str, Any] = field(default_factory=dict)  # latest evidence: attempt, status, freshness, at
    history: list[dict[str, Any]] = field(default_factory=list)  # bounded: at, attempt, event, status, detail

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def add_history(self, event: str, attempt: int, detail: str = "") -> None:
        self.history = (self.history + [{"at": time.time(), "attempt": attempt, "event": event, "status": self.status,
                                         "detail": _s(detail, 300)}])[-MAX_CRITERION_HISTORY:]

    @classmethod
    def from_any(cls, raw: Any) -> "HumanCriterion | None":
        if not isinstance(raw, dict) or not _s(raw.get("text")) or not _s(raw.get("id")):
            return None
        status = _s(raw.get("status"), 20)
        return cls(id=_s(raw["id"], 40), text=_s(raw["text"], 600), verdict=_s(raw.get("verdict"), 20),
                   note=_s(raw.get("note"), 600), attempt=int(raw.get("attempt") or 0),
                   created_at=float(raw.get("created_at") or 0.0), kind=_token(raw.get("kind")),
                   rationale=_s(raw.get("rationale"), 600), source=_s(raw.get("source"), 40) or "human_followup",
                   request_id=_s(raw.get("request_id"), 60), feedback_ref=_s(raw.get("feedback_ref"), 100),
                   feedback_refs=_strs(raw.get("feedback_refs"), 10, 100), reports=max(1, int(raw.get("reports") or 1)),
                   status=status if status in CRITERION_STATUSES else "open",
                   evidence=dict(raw.get("evidence") or {}) if isinstance(raw.get("evidence"), dict) else {},
                   history=[x for x in raw.get("history") or [] if isinstance(x, dict)][-MAX_CRITERION_HISTORY:])


def _token(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v or "").strip().lower()).strip("_")[:40]


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", str(text or "").lower()).strip()


# Words that carry no meaning of their own in a criterion. Deliberately small: negations, numbers,
# prepositions and every content word stay, so genuinely different criteria never compare equal.
_FILLER = frozenset({"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "should", "must", "shall",
                     "will", "would", "can", "could", "does", "do", "did", "has", "have", "had"})


def _stem(w: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not w.endswith("ss"):
            return w[:-len(suffix)]
    return w


_CONTRACTIONS = ((re.compile(r"n['’]t\b", re.IGNORECASE), " not"),
                 (re.compile(r"['’](s|re|ll|ve|d|m)\b", re.IGNORECASE), ""))


def criterion_key(text: str) -> tuple[str, ...]:
    """Deterministic normalization for duplicate detection: case, punctuation, possessives/contractions,
    filler words and simple inflection are ignored; word ORDER and every content word are kept ("rear
    wheels" != "front wheels", "A before B" != "B before A", "not" / "n't" are never dropped)."""
    t = str(text or "")
    for rx, repl in _CONTRACTIONS:
        t = rx.sub(repl, t)
    return tuple(_stem(w) for w in _norm(t).split() if w not in _FILLER)


def merge_derived_criteria(existing: list[HumanCriterion], phase_id: str, raw: Any, feedback: HumanFeedback | None,
                           attempt: int, request_id: str = "") -> tuple[list[HumanCriterion], list[HumanCriterion]]:
    """Add Claude's derived criteria to `existing` as HF-<phase>-NN. A criterion that normalizes to an existing
    one (the same issue reported again) is NOT duplicated: the existing criterion records the new report.
    Returns (added, reported again)."""
    keys = {criterion_key(c.text): c for c in existing}
    n = max([int(m.group(1)) for c in existing if (m := re.search(r"-(\d+)$", c.id))] or [0])
    ref = f"{request_id or (feedback.request_id if feedback else '')}@{feedback.created_at if feedback else 0:.3f}"
    added: list[HumanCriterion] = []
    again: list[HumanCriterion] = []
    for x in raw if isinstance(raw, list) else []:
        text = _s(x.get("text") if isinstance(x, dict) else x, 600)
        key = criterion_key(text)
        if not text or not key:
            continue
        same = keys.get(key)
        if same is not None:
            if ref not in same.feedback_refs and same not in added:
                same.reports += 1
                same.feedback_refs = (same.feedback_refs + [ref])[-10:]
                same.add_history("reported again", attempt, f"{feedback.verdict if feedback else ''}: "
                                                            f"{feedback.note if feedback else ''}")
                again.append(same)
            continue
        if len(existing) >= MAX_CRITERIA:
            continue
        n += 1
        c = HumanCriterion(f"HF-{phase_id}-{n:02d}", text, feedback.verdict if feedback else "",
                           _s(feedback.note, 600) if feedback else "", attempt, time.time(),
                           kind=_token(x.get("kind") if isinstance(x, dict) else ""),
                           rationale=_s(x.get("rationale") if isinstance(x, dict) else "", 600),
                           request_id=request_id or (feedback.request_id if feedback else ""), feedback_ref=ref,
                           feedback_refs=[ref])
        c.add_history("derived", attempt, f"from the user's {c.verdict or 'answer'}")
        existing.append(c)
        added.append(c)
        keys[key] = c
    return added, again


def add_derived_criteria(existing: list[HumanCriterion], phase_id: str, raw: Any, feedback: HumanFeedback | None,
                         attempt: int) -> list[HumanCriterion]:
    """Append Claude's derived criteria to `existing` (duplicates are skipped) with ids HF-<phase>-NN.
    Returns the criteria that were added."""
    return merge_derived_criteria(existing, phase_id, raw, feedback, attempt)[0]


def record_criterion_evidence(c: HumanCriterion, result: dict[str, Any], attempt: int) -> bool:
    """Associate a criterion with its latest evidence result. Returns True if its status changed."""
    status = "verified" if is_satisfied(result) else ("failed" if result.get("status") == FAILED else "unverified")
    c.evidence = {"attempt": attempt, "status": result.get("status"), "freshness": result.get("freshness", ""),
                  "summary": _s(result.get("summary") or result.get("failure_reason"), 200), "at": time.time()}
    changed = status != c.status
    c.status = status
    if changed:
        c.add_history("evidence", attempt, f"{result.get('status')}: {c.evidence['summary']}")
    return changed


def criteria_lines(criteria: list[HumanCriterion], with_status: bool = False) -> list[str]:
    if not with_status:
        return [f"{c.id}: {c.text}" for c in criteria]
    out = []
    for c in criteria:
        ev = c.evidence or {}
        tail = f" [{c.status}" + (f": evidence {ev.get('status')} in attempt {ev.get('attempt')}" if ev else "") + "]"
        out.append(f"{c.id}: {c.text}" + (tail if c.status != "open" or ev else " [open: no evidence yet]"))
    return out


def evidence_summary(verification: list[Any], assessment: list[dict[str, Any]], notes: list[str],
                     bundle: dict[str, Any] | None = None) -> list[str]:
    """Compact automated evidence at the moment of an ASK_USER request: the evidence results first
    (✓ satisfied, ✗ failed, ? unavailable, ~ inconclusive), then the verification commands."""
    lines = [result_line(r, 160) for r in (bundle or {}).get("results") or []]
    for v in verification[:8]:
        if v.skipped_reason:
            st = f"skipped ({short(v.skipped_reason, 80)})"
        else:
            st = "passed" if v.passed else ("timed out" if v.exit_code is None else f"failed (exit {v.exit_code})")
        lines.append(f"verification `{short(v.command, 120)}`: {st}")
    if not verification:
        lines.append("no verification commands ran for this phase")
    for a in assessment[:8]:
        lines.append(f"evidence {a.get('id')}: {a.get('status') or '?'}" + (f" - {short(a['note'], 160)}"
                                                                             if a.get("note") else ""))
    lines += [f"note: {short(x, 200)}" for x in notes[:4]]
    return lines[:MAX_EVIDENCE_LINES]


def select_questions(raw_questions: Any, raw_targets: Any, results: list[dict[str, Any]], known_ids: set[str],
                     max_questions: int = MAX_QUESTIONS) -> tuple[list[str], list[str], list[str]]:
    """(questions, targets, dropped): Claude's questions, deduplicated and bounded. A question Claude tied to
    a requirement that automation already proved (a fresh SATISFIED result) is not asked - unless nothing
    would be left to ask. Nothing is parsed from the question text: only Claude's structured targets count."""
    qs = [str(q).strip()[:400] for q in raw_questions if str(q or "").strip()] if isinstance(raw_questions, list) else []
    ts = [str(t or "").strip()[:40] for t in raw_targets] if isinstance(raw_targets, list) else []
    ts = [t if t in known_ids else "" for t in ts] + [""] * max(0, len(qs) - len(ts))
    proven = {str(r.get("requirement_id")) for r in results if is_satisfied(r)}
    keep: list[tuple[str, str]] = []
    dropped: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for q, t in zip(qs, ts):
        key = criterion_key(q)
        if key in seen:
            dropped.append(f"duplicate: {short(q, 200)}")
            continue
        seen.add(key)
        keep.append((q, t))
    if any(t not in proven for _q, t in keep):
        for q, t in [x for x in keep if x[1] and x[1] in proven]:
            dropped.append(f"already established by automation ({t}): {short(q, 200)}")
            keep.remove((q, t))
    limit = max(1, min(int(max_questions or MAX_QUESTIONS), MAX_QUESTIONS))
    dropped += [f"over the limit of {limit} question(s): {short(q, 200)}" for q, _t in keep[limit:]]
    keep = keep[:limit]
    targets = [t for _q, t in keep]
    return [q for q, _t in keep], (targets if any(targets) else []), dropped


def build_request(phase: Any, attempt: int, review: Any, final: bool, verification: list[Any],
                  evidence: dict[str, Any] | None = None, max_questions: int = MAX_QUESTIONS) -> HumanRequest:
    """The persisted ASK_USER request: Claude's payload, the contract's human checklist, what the automated
    evidence established (AUTOMATION ESTABLISHED), what it could not (AUTOMATION COULD NOT ESTABLISH) and the
    questions to check (PLEASE CHECK: specific, deduplicated, bounded)."""
    hr = review.human_request or {}
    contract = getattr(phase, "verification_contract", None)
    checklist = _strs((list(contract.human_gate.checklist) if contract else []) + list(hr.get("checklist") or []),
                      MAX_CHECKLIST)
    results = list((evidence or {}).get("results") or [])
    missing = _strs(getattr(review, "missing_evidence", []), MAX_OBSERVATIONS) or [
        f"{r.get('requirement_id')} [{r.get('kind')}]: {short(r.get('goal') or r.get('summary') or '', 160)}"
        for r in results if r.get("required") and not is_satisfied(r)
    ][:MAX_OBSERVATIONS]
    established = [f"{r.get('requirement_id')} [{r.get('kind')}]: {short(r.get('summary') or r.get('goal') or '', 160)}"
                   for r in results if is_satisfied(r)][:MAX_OBSERVATIONS]
    known = {str(r.get("requirement_id")) for r in results}
    reqs = getattr(phase, "evidence_requirements", None)
    if callable(reqs):
        known |= {r.id for r in reqs()}
    questions, targets, dropped = select_questions(hr.get("questions"), hr.get("question_targets"), results, known,
                                                   max_questions)
    return HumanRequest(
        phase_id=phase.id, reason=_s(hr.get("reason") or review.summary), questions=questions,
        observations=_strs(hr.get("observations"), MAX_OBSERVATIONS), checklist=checklist,
        evidence=evidence_summary(verification, review.evidence_assessment, review.verification_notes, evidence),
        phase_title=phase.title, attempt=attempt, checkpoint_summary=_s(review.summary, 2000), final=final,
        created_at=time.time(), missing=missing, established=established, targets=targets, dropped=dropped)
