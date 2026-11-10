"""V4: the human verification panel. Shown while Claude waits for the user to verify a phase.

It renders the persisted request (phase, Claude's reason, observations, automated evidence, the
generated questions / checklist) and collects a STRUCTURED answer: one of three verdicts plus a
free-text note that is passed on as written (never parsed).
"""
from __future__ import annotations

import html
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QTextBrowser, QVBoxLayout,
)

VERDICT_BUTTONS = (("LOOKS_GOOD", "Looks good"), ("ISSUE_FOUND", "Issue found"), ("CANNOT_VERIFY", "Cannot verify"))


def _e(s: Any) -> str:
    return html.escape(str(s))


def _items(title: str, items: list[Any], ordered: bool = False) -> str:
    if not items:
        return ""
    tag = "ol" if ordered else "ul"
    return f"<b>{_e(title)}</b><{tag}>" + "".join(f"<li>{_e(x)}</li>" for x in items) + f"</{tag}>"


class HumanGatePanel(QGroupBox):
    feedback_submitted = Signal(str, str)  # verdict, note

    def __init__(self, parent=None):
        super().__init__("Human verification requested", parent)
        self.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #07a; border-radius: 4px; margin-top: 8px; }"
                           "QGroupBox::title { color: #07a; subcontrol-origin: margin; left: 8px; }")
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        self.view.setMinimumHeight(150)
        self.note = QPlainTextEdit()
        self.note.setPlaceholderText("What did you observe? (required for 'Issue found'; sent to Claude as written)")
        self.note.setFixedHeight(70)
        self.buttons: dict[str, QPushButton] = {}
        row = QHBoxLayout()
        for verdict, text in VERDICT_BUTTONS:
            b = QPushButton(text)
            b.setMinimumHeight(30)
            b.clicked.connect(lambda _=False, v=verdict: self._submit(v))
            self.buttons[verdict] = b
            row.addWidget(b)
        row.addStretch(1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-weight: normal; color: #555;")
        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.note)
        lay.addLayout(row)
        lay.addWidget(self.status)
        self.human: dict[str, Any] | None = None
        self._request_id = ""
        self.setVisible(False)

    def set_request(self, human: dict[str, Any] | None, state: str) -> None:
        self.human = human
        if not human or not human.get("request"):
            self._request_id = ""
            self.setVisible(False)
            return
        req = human["request"]
        rid = f"{req.get('phase_id')}-a{req.get('attempt')}-{req.get('created_at')}"
        if rid != self._request_id:  # a new request: fresh note
            self._request_id = rid
            self.note.clear()
        title = req.get("phase_title") or ""
        self.setTitle(f"Human verification requested — {req.get('phase_id')}" + (f" · {title}" if title else ""))
        targets = list(req.get("targets") or [])
        questions = [(f"[{targets[i]}] " if i < len(targets) and targets[i] else "") + str(q)
                     for i, q in enumerate(req.get("questions") or [])]
        established = list(req.get("established") or []) or list(req.get("observations") or [])
        self.view.setHtml(
            f"<p><b>Claude:</b> {_e(req.get('reason', ''))}</p>"
            + _items("PLEASE CHECK in the running result:", questions, ordered=True)
            + _items("AUTOMATION COULD NOT ESTABLISH", req.get("missing") or [])
            + _items("AUTOMATION ESTABLISHED", established)
            + _items("Checklist", req.get("checklist") or [])
            + (_items("Also noted by Claude", req.get("observations") or []) if req.get("established") else "")
            + _items("Automated evidence", req.get("evidence") or [])
            + (f"<p style='color:#666'>Checkpoint: {_e(req.get('checkpoint_summary', ''))}</p>"
               if req.get("checkpoint_summary") else ""))
        waiting = human.get("status") == "awaiting"
        answerable = waiting and state == "AWAITING_HUMAN"
        for b in self.buttons.values():
            b.setEnabled(answerable)
        self.note.setReadOnly(not answerable)
        fb = human.get("feedback") or {}
        if waiting and state != "AWAITING_HUMAN":
            self.status.setText(f"The run is {state}; RESUME to answer this request.")
        elif waiting:
            self.status.setText("Nothing runs until you answer. Your answer is adjudicated by Claude (HUMAN_FOLLOWUP).")
        else:
            if fb.get("note") and not self.note.toPlainText():
                self.note.setPlainText(fb["note"])
            self.status.setText(f"Answer recorded: {fb.get('verdict', '?')}. Claude HUMAN_FOLLOWUP adjudicates it "
                                "(if it was interrupted, RESUME runs it again).")
        self.setVisible(True)

    def _submit(self, verdict: str) -> None:
        note = self.note.toPlainText().strip()
        if verdict == "ISSUE_FOUND" and not note:
            QMessageBox.information(self, "Issue found", "Describe what you observed in the note first.")
            return
        for b in self.buttons.values():
            b.setEnabled(False)
        self.note.setReadOnly(True)
        self.status.setText(f"Sending {verdict}…")
        self.feedback_submitted.emit(verdict, note)
