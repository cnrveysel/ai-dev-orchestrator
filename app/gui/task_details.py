"""Right panel: details for the selected task and its attempts."""
from __future__ import annotations

import html
import time
from typing import Any

from PySide6.QtWidgets import QTextBrowser


def _e(s: Any) -> str:
    return html.escape(str(s))


def _ul(items: list[Any]) -> str:
    if not items:
        return "<p style='color:#888'>(none)</p>"
    return "<ul>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"


def _pre(s: str) -> str:
    return f"<pre style='background:#f4f4f4;padding:4px;white-space:pre-wrap'>{_e(s)}</pre>" if s else ""


class TaskDetails(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.show_plan_overview(None)

    def show_plan_overview(self, plan: dict[str, Any] | None) -> None:
        if not plan:
            self.setHtml("<h3>No plan yet</h3><p>Enter a request, press <b>PLAN</b>, review the plan, "
                         "then press <b>START</b>.</p>")
            return
        self.setHtml(
            f"<h3>Plan</h3><p><b>Goal:</b> {_e(plan.get('goal', ''))}</p>"
            f"<p>{_e(plan.get('summary', ''))}</p>"
            f"<h4>Assumptions</h4>{_ul(plan.get('assumptions', []))}"
            f"<h4>Architecture notes</h4>{_ul(plan.get('architecture_notes', []))}"
            f"<p style='color:#888'>Select a task to see its details.</p>")

    def show_task(self, t: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
        parts = [
            f"<h3>{_e(t['id'])} — {_e(t.get('title', ''))}</h3>",
            f"<p><b>Status:</b> {_e(t.get('status'))} &nbsp; <b>Attempts:</b> {t.get('attempts', 0)}"
            + (f" &nbsp; <b>Depends on:</b> {_e(', '.join(t['dependencies']))}" if t.get("dependencies") else "")
            + "</p>",
            f"<h4>Description</h4><p>{_e(t.get('description', '')).replace(chr(10), '<br>')}</p>",
            f"<h4>Acceptance criteria</h4>{_ul(t.get('acceptance_criteria', []))}",
            f"<h4>Claude instructions</h4>{_pre(t.get('instructions', '')) or '<p>(none)</p>'}",
            f"<h4>Files likely involved</h4>{_ul(t.get('files_likely_involved', []))}",
            f"<h4>Verification commands</h4>{_ul(t.get('verification_commands', []))}",
        ]
        if t.get("kind") == "phase":
            parts.append(self._phase_html(t))
            parts.append(self._evidence_html(t, attempts))
        if t.get("user_guidance"):
            parts.append(f"<h4>User guidance</h4>{_pre(t['user_guidance'])}")
        res = t.get("result")
        if res:
            parts.append("<h4>Compact result (passed to later tasks)</h4>"
                         f"<p>{_e(res.get('result', ''))}</p>"
                         f"<b>Decisions</b>{_ul(res.get('important_decisions', []))}"
                         f"<b>Risks</b>{_ul(res.get('remaining_risks', []))}")
        if attempts:
            latest = attempts[-1]
            worker_files = [f"{c['change']} {c['path']}" + (f"  ({c['note']})" if c.get("note") else "")
                            for c in latest.get("changed_files", [])]
            scope = ("whole phase so far" if latest.get("diff_scope") == "phase" else "latest attempt")
            parts.append(f"<h4>Files actually changed ({scope}, by worker only)</h4>{_ul(worker_files)}")
            if latest.get("preexisting_dirty"):
                parts.append(f"<p style='color:#888'>Pre-existing user changes (not attributed): "
                             f"{len(latest['preexisting_dirty'])}</p>")
        for a in reversed(attempts):
            parts.append(self._attempt_html(a))
        self.setHtml("".join(parts))

    def _phase_html(self, t: dict[str, Any]) -> str:
        subs = t.get("subtasks") or []
        rows = "".join(f"<li>{'☑' if x.get('done') else '☐'} <b>{_e(x.get('id'))}</b> {_e(x.get('title', ''))}"
                       + (f" — <span style='color:#666'>{_e(x.get('description'))}</span>" if x.get("description") else "")
                       + "</li>" for x in subs)
        html_parts = [f"<h4>Subtasks (worker-managed, {sum(1 for x in subs if x.get('done'))}/{len(subs)} reported done)</h4>"
                      f"<ul>{rows}</ul>" if subs else ""]
        if t.get("integration_expectations"):
            html_parts.append(f"<h4>Integration expectations</h4>{_ul(t['integration_expectations'])}")
        session = t.get("session_id") or "(none yet)"
        html_parts.append(f"<p><b>Qwen session:</b> {_e(session)} &nbsp; <b>turns in session:</b> {t.get('session_turns', 0)}"
                          f" &nbsp; <b>sessions started:</b> {t.get('sessions_started', 0)}"
                          + (f" &nbsp; <b>worker status:</b> {_e(t['reported_status'])}" if t.get("reported_status") else "")
                          + (f" &nbsp; <b>last checkpoint:</b> {_e(t['last_checkpoint'])}" if t.get("last_checkpoint") else "")
                          + (" &nbsp; <b style='color:#b60'>fresh session requested</b>" if t.get("fresh_session_requested") else "")
                          + (f" &nbsp; <b>estimate:</b> ~{t['estimated_minutes']} min" if t.get("estimated_minutes") else "")
                          + "</p>")
        todo = t.get("todo") or []
        if todo:
            html_parts.append("<h4>Qwen todo list (live)</h4><ul>" + "".join(
                f"<li>{'☑' if x.get('completed') else '☐'} {_e(x.get('text', ''))}</li>" for x in todo) + "</ul>")
        if t.get("human_criteria"):
            def crit(c: dict[str, Any]) -> str:
                ev = c.get("evidence") or {}
                extra = [c.get("status") or "open"]
                if ev:
                    extra.append(f"evidence {ev.get('status')} (attempt {ev.get('attempt')})")
                if int(c.get("reports") or 1) > 1:
                    extra.append(f"reported {c.get('reports')}×")
                return f"{c.get('id')}: {c.get('text')} — {', '.join(extra)}"
            html_parts.append("<h4>Human-derived acceptance criteria (evidence requirements)</h4>" + _ul(
                [crit(c) for c in t["human_criteria"]]))
        c = t.get("verification_contract")
        if c:
            reqs = [f"{r.get('id')} [{r.get('kind')}{', required' if r.get('required') else ', optional'}] "
                    f"{r.get('goal')}" + (f" — {r.get('reason')}" if r.get("reason") else "")
                    for r in c.get("evidence_requirements") or []]
            gate = c.get("human_gate") or {}
            html_parts.append("<h4>Verification contract</h4>"
                              + (f"<p><b>Success dimensions:</b> {_e(', '.join(c.get('success_dimensions') or []))}</p>"
                                 if c.get("success_dimensions") else "")
                              + f"<b>Evidence requirements</b>{_ul(reqs)}"
                              + f"<p><b>Human gate:</b> {'allowed' if gate.get('allowed') else 'not allowed'}"
                              + (f" — {_e(gate.get('reason'))}" if gate.get("reason") else "") + "</p>"
                              + (f"<b>Human checklist</b>{_ul(gate['checklist'])}" if gate.get("checklist") else ""))
        if t.get("superseded_reason"):
            html_parts.append(f"<p style='color:#b60'><b>Superseded by replan:</b> {_e(t['superseded_reason'])}</p>")
        return "".join(html_parts)

    def _attempt_html(self, a: dict[str, Any]) -> str:
        started = time.strftime("%H:%M:%S", time.localtime(a.get("started_at") or 0))
        elapsed = (a["finished_at"] - a["started_at"]) if a.get("finished_at") else None
        head = (f"<hr><h4>Attempt {a.get('number')} ({_e(a.get('kind'))}) — started {started}"
                + (f", {elapsed:.0f}s" if elapsed is not None else ", running…")
                + (" — <span style='color:#b00'>INTERRUPTED</span>" if a.get("interrupted") else "") + "</h4>")
        body = [f"<p><b>Executable:</b> {_e(a.get('executable') or '(pending)')}<br>"
                f"<b>Exit code:</b> {_e(a.get('exit_code'))} &nbsp; <b>Tool calls:</b> {a.get('tool_calls', 0)}</p>"]
        if a.get("error"):
            body.append(f"<p style='color:#b00'><b>Error:</b> {_e(a['error'])}</p>")
        turns = a.get("turns") or []
        if turns:
            rows = "".join(
                f"<tr><td>{x.get('n')}</td><td>{_e(x.get('purpose'))}</td><td>{_e(x.get('mode'))}</td>"
                f"<td>{'running…' if x.get('running') else _e(x.get('exit_code'))}</td>"
                f"<td>{_e(x.get('duration', ''))}s</td><td>{_e(x.get('tool_calls', ''))}</td>"
                f"<td>{_e(x.get('reported_status', ''))}</td></tr>" for x in turns)
            body.append("<b>Qwen turns</b><table cellspacing=4><tr><th>#</th><th>purpose</th><th>session</th>"
                        f"<th>exit</th><th>time</th><th>tools</th><th>status</th></tr>{rows}</table>"
                        + (f"<p>Local verification fix rounds: {a['local_fix_rounds']}</p>" if a.get("local_fix_rounds") else ""))
        if a.get("worker_summary"):
            body.append(f"<b>Worker summary</b>{_pre(a['worker_summary'])}")
        v = a.get("verification") or []
        if v:
            rows = "".join(
                f"<tr><td>{'✅' if x.get('passed') else ('⏭️' if x.get('skipped_reason') else '❌')}</td>"
                f"<td><code>{_e(x.get('command'))}</code></td>"
                f"<td>{_e(x.get('skipped_reason') or ('exit ' + str(x.get('exit_code'))))}</td></tr>" for x in v)
            body.append(f"<b>Verification</b><table cellspacing=4>{rows}</table>")
            for x in v:
                if not x.get("passed") and x.get("output_tail"):
                    body.append(_pre(x["output_tail"][-2000:]))
        r = a.get("review")
        if r:
            color = {"PASS": "#080", "RETRY": "#b60", "BLOCKED": "#b00", "REPLAN": "#70a",
                     "ASK_USER": "#07a"}.get(r.get("decision"), "#000")
            what = {"CHECKPOINT": "Claude checkpoint", "FINAL_AUDIT": "Claude final audit",
                    "HUMAN_FOLLOWUP": "Claude human follow-up",
                    "CONSISTENCY_REVIEW": "Claude consistency review"}.get(r.get("kind"), "Claude review")
            body.append(f"<b>{what}: <span style='color:{color}'>{_e(r.get('decision'))}</span></b>"
                        f"<p>{_e(r.get('summary', ''))}</p>"
                        + (f"<b>Problems</b>{_ul(r['problems'])}" if r.get("problems") else "")
                        + (f"<b>Corrective instructions</b>{_ul(r['corrective_instructions'])}"
                           if r.get("corrective_instructions") else "")
                        + (f"<b>Verification notes</b>{_ul(r['verification_notes'])}"
                           if r.get("verification_notes") else "")
                        + (f"<b>Integration notes</b>{_ul(r['integration_notes'])}" if r.get("integration_notes") else "")
                        + (f"<b>Regressions</b>{_ul(r['regressions'])}" if r.get("regressions") else "")
                        + (f"<b>Remaining risks</b>{_ul(r['remaining_risks'])}" if r.get("remaining_risks") else "")
                        + (f"<b>Replan reason</b><p>{_e(r['replan_reason'])}</p>" if r.get("replan_reason") else ""))
            body.append(self._human_html(r))
        if a.get("diff_stat"):
            scope = "whole phase so far" if a.get("diff_scope") == "phase" else "this attempt only"
            body.append(f"<b>Git diff summary ({scope})</b>{_pre(a['diff_stat'])}")
        return head + "".join(body)

    @staticmethod
    def _evidence_html(t: dict[str, Any], attempts: list[dict[str, Any]]) -> str:
        """V4 pass 2: per-requirement evidence of the latest attempt that collected evidence (compact)."""
        wanted = bool((t.get("verification_contract") or {}).get("evidence_requirements"))
        a = next((x for x in reversed(attempts) if x.get("evidence")), None)
        if a is None:
            return "<h4>Evidence</h4><p style='color:#888'>(not collected yet)</p>" if wanted else ""
        ev = a["evidence"]
        colors = {"SATISFIED": "#080", "FAILED": "#b00", "UNAVAILABLE": "#b60", "INCONCLUSIVE": "#b60",
                  "SKIPPED": "#888"}

        def fresh(r: dict[str, Any]) -> str:
            f = r.get("freshness") or ""
            if f == "stale":
                return f"<b style='color:#b00'>STALE</b> (was {_e(r.get('stale_status') or '?')})"
            if f == "reused":
                return f"reused (attempt {_e((r.get('provenance') or {}).get('reused_from', '?'))})"
            return _e(f)

        rows = "".join(
            f"<tr><td><b style='color:{colors.get(r.get('status'), '#000')}'>{_e(r.get('status'))}</b></td>"
            f"<td>{_e(r.get('requirement_id'))}{' 🙋' if r.get('human_derived') else ''}</td>"
            f"<td>{_e(r.get('kind'))}{'' if r.get('required') else ' (optional)'}</td><td>{fresh(r)}</td>"
            f"<td>{_e(', '.join(r.get('capability_ids') or []) or r.get('strategy'))}</td>"
            f"<td>{_e((r.get('stale_reason') or r.get('summary') or r.get('failure_reason') or '')[:220])}"
            + (f" <span style='color:#888'>[{_e(r.get('cause'))}]</span>" if r.get("cause") else "")
            + "".join(f"<br><code>{_e(x)}</code>" for x in (r.get("diagnostics") or [])[:3]) + "</td></tr>"
            for r in ev.get("results") or [])
        review = a.get("review") or {}
        judged = review.get("evidence_sufficient")
        verdict = ("<p><b>Claude:</b> evidence " + ("<b style='color:#080'>sufficient</b>" if judged else
                                                    "<b style='color:#b00'>not sufficient</b>") + "</p>"
                   if judged is not None else "<p style='color:#888'>(not judged by a checkpoint yet)</p>")
        cons = review.get("consistency") or {}
        if review.get("kind") == "CONSISTENCY_REVIEW" and cons:
            verdict += ("<p style='color:#70a'><b>Consistency review</b> (the checkpoint contradicted its evidence "
                        f"judgment: {_e((cons.get('trigger') or {}).get('reason', ''))}): "
                        f"<b>{_e(review.get('decision'))}</b>"
                        + (f" — {_e(cons.get('justification'))}" if cons.get("justification") else "") + "</p>"
                        + _ul([f"{d.get('requirement_id')}: {d.get('disposition')} — {d.get('reason')}"
                               for d in cons.get("dispositions") or []]))
        missing = review.get("missing_evidence") or [
            f"{r.get('requirement_id')} ({r.get('kind')}): {r.get('status')}" for r in ev.get("results") or []
            if r.get("required") and (r.get("status") != "SATISFIED" or r.get("freshness") == "stale")]
        caps = [f"{c.get('id')} [{c.get('kind')}] {c.get('name')} — {c.get('confidence')}"
                + (f" (via {(c.get('provenance') or {}).get('discovered_by')})"
                   if (c.get("provenance") or {}).get("discovered_by") else "")
                + (" — declared in a file changed in this phase"
                   if (c.get("provenance") or {}).get("changed_in_phase") else "")
                for c in ev.get("capabilities") or []]
        arts = [f"{x.get('id')} [{x.get('kind')}{', ' + x['media_type'] if x.get('media_type') else ''}] "
                f"{x.get('description')} — {x.get('path_or_ref')}" for x in ev.get("artifacts") or []]
        replans = [f"replan {x.get('generation')}: {', '.join(x.get('requirements') or [])} — "
                   + "; ".join(str(r) for r in (x.get("reasons") or [])[:2]) for x in ev.get("replans") or []]
        guard = (ev.get("guard") or {}).get("detail", "")
        return (f"<h4>Evidence (attempt {a.get('number')})</h4>"
                f"<table cellspacing=4><tr><th>status</th><th>req.</th><th>kind</th><th>freshness</th>"
                f"<th>capabilities</th><th>summary</th></tr>{rows}</table>{verdict}"
                + (f"<b>Missing evidence</b>{_ul(missing)}" if missing else "")
                + (f"<b>Replans</b>{_ul(replans)}" if replans else "")
                + f"<p style='color:#666'>Discovery: {_e(ev.get('capability_summary', ''))}; plan: "
                  f"{_e(ev.get('plan_source', ''))}; {ev.get('steps_run', 0)} step(s), {ev.get('wall_used', 0)}s"
                + (f"; repository protection: {_e(guard)}" if guard else "") + "</p>"
                + (f"<b>Capabilities used</b>{_ul(caps)}" if caps else "")
                + (f"<b>Artifacts</b>{_ul(arts[:12])}" if arts else "")
                + (f"<b>Notes</b>{_ul(ev['notes'][:6])}" if ev.get("notes") else ""))

    @staticmethod
    def _human_html(r: dict[str, Any]) -> str:
        """V4: the human verification request / answer / derived criteria attached to a review."""
        req, fb = r.get("human_request") or {}, r.get("human_feedback") or {}
        parts = []
        if req.get("questions"):
            parts.append("<p style='color:#07a'><b>Human verification requested:</b> "
                         f"{_e(req.get('reason', ''))}</p><ol>"
                         + "".join(f"<li>{_e(q)}</li>" for q in req["questions"]) + "</ol>")
        if fb:
            parts.append(f"<p><b>User's answer:</b> {_e(fb.get('verdict'))}"
                         + (f" — “{_e(fb.get('note'))}”" if fb.get("note") else "") + "</p>")
        if r.get("derived_criteria"):
            parts.append("<b>Derived human criteria</b>" + _ul(
                [f"{c.get('id', '')}: {c.get('text')}".lstrip(": ") for c in r["derived_criteria"]]))
        if r.get("do_not_redo"):
            parts.append(f"<b>Do not redo</b>{_ul(r['do_not_redo'])}")
        if r.get("issues"):
            parts.append("<b>Issues</b>" + _ul(
                [("[" + "/".join(x for x in (i.get("severity"), i.get("disposition")) if x) + "] "
                  if i.get("severity") or i.get("disposition") else "") + str(i.get("description", ""))
                 for i in r["issues"]]))
        if r.get("human_request", {}).get("missing"):
            parts.append(f"<b>Not established automatically</b>{_ul(r['human_request']['missing'])}")
        if r.get("evidence_assessment"):
            parts.append("<b>Evidence assessment</b>" + _ul(
                [f"{x.get('id')}: {x.get('status')}" + (f" — {x.get('note')}" if x.get("note") else "")
                 for x in r["evidence_assessment"]]))
        return "".join(parts)
