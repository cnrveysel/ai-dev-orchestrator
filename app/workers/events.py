"""V5.1 normalized worker events.

Each runtime adapter maps its CLI's native stream (Codex `exec --json`, Cline `--json`) to WorkerEvent, so logging,
guards and future consumers see one vocabulary. The schedulers never see native events.

    session_started  a session / thread id became known
    assistant_text   a complete assistant message (streamed chunks are not separate events)
    reasoning        a complete reasoning/thinking block
    tool_started     a tool call started (tool_id is unique per call)
    tool_finished    a tool call ended (text = output tail, error set on failure)
    command_output   partial tool output while it runs
    usage            token usage (raw carries the numbers)
    warning          a notice / recoverable problem
    error            an error reported by the worker
    completed        the turn ended (text = final message, raw carries the finish reason)

`raw` keeps the native event for diagnostics, with secret-looking keys redacted.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.workers.profiles import REDACTED, is_secret_name

KINDS = ("session_started", "assistant_text", "reasoning", "tool_started", "tool_finished", "command_output", "usage",
         "warning", "error", "completed")
_PATCH_FILE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+?)\s*$", re.M)


@dataclass
class WorkerEvent:
    kind: str
    text: str = ""
    tool_name: str = ""
    tool_id: str = ""
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    error: str = ""
    files: list[tuple[str, str]] = field(default_factory=list)  # (kind add|update|delete, path) of a file change
    raw: dict[str, Any] = field(default_factory=dict)


def redact(obj: Any, key: str = "") -> Any:
    """Deep copy with values under secret-looking keys replaced (for diagnostics / debug logs)."""
    if isinstance(obj, dict):  # credentials are strings; counts such as inputTokens are not secrets
        return {k: (REDACTED if is_secret_name(str(k)) and isinstance(v, str) and v else redact(v, str(k)))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, key) for v in obj]
    return obj


def _text(v: Any, limit: int = 4000) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v[:limit]
    try:
        return json.dumps(v, ensure_ascii=False)[:limit]
    except (TypeError, ValueError):
        return str(v)[:limit]


def patch_files(patch: str) -> list[tuple[str, str]]:
    """File changes named by an apply_patch body (`*** Add|Update|Delete File: <path>`)."""
    return [(m.group(1).lower(), m.group(2)) for m in _PATCH_FILE.finditer(patch or "")]


def tool_file_changes(tool_name: str, tool_input: Any) -> list[tuple[str, str]]:
    """Cline file-editing tools: editor {path, ...} and apply_patch {input: <patch>}."""
    if tool_name == "editor" and isinstance(tool_input, dict) and tool_input.get("path"):
        return [("update", str(tool_input["path"]))]
    if tool_name == "apply_patch":
        body = tool_input.get("input") if isinstance(tool_input, dict) else tool_input
        return patch_files(body if isinstance(body, str) else "")
    return []


def normalize_cline_event(ev: dict[str, Any]) -> list[WorkerEvent]:
    """One Cline CLI `--json` line (already parsed) -> normalized events. Streaming text/reasoning chunks
    (content_start of text/reasoning) are not events: their complete block arrives with content_end."""
    etype = ev.get("type")
    raw = redact(ev)
    if etype == "agent_event":
        e = ev.get("event") or {}
        r = raw.get("event") or {}  # text shown to people is always rendered from the redacted copy
        t, ctype = e.get("type"), e.get("contentType")
        if t == "content_start" and ctype == "tool":
            name = str(e.get("toolName") or "tool")
            return [WorkerEvent("tool_started", text=_text(r.get("input"), 400), tool_name=name,
                                tool_id=str(e.get("toolCallId") or ""), files=tool_file_changes(name, e.get("input")),
                                raw=raw)]
        if t == "content_update" and ctype == "tool":
            return [WorkerEvent("command_output", text=_text(r.get("update"), 1500), tool_name=str(e.get("toolName") or ""),
                                tool_id=str(e.get("toolCallId") or ""), raw=raw)]
        if t == "content_end" and ctype == "tool":
            return [WorkerEvent("tool_finished", text=_text(r.get("output"), 1500), tool_name=str(e.get("toolName") or ""),
                                tool_id=str(e.get("toolCallId") or ""), error=str(e.get("error") or ""), raw=raw)]
        if t == "content_end" and ctype == "text" and e.get("text"):
            return [WorkerEvent("assistant_text", text=str(e["text"]), raw=raw)]
        if t == "content_end" and ctype == "reasoning" and e.get("reasoning"):
            return [WorkerEvent("reasoning", text=str(e["reasoning"]), raw=raw)]
        if t == "usage":
            return [WorkerEvent("usage", raw=raw)]
        if t == "notice":
            return [WorkerEvent("warning", text=str(e.get("message") or ""), raw=raw)]
        if t == "error":
            err = e.get("error")
            msg = str((err.get("message") if isinstance(err, dict) else err) or "error")
            if e.get("recoverable"):  # e.g. a rejected tool call recorded as a "consecutive mistake": the agent goes on
                return [WorkerEvent("warning", text=f"recoverable: {msg}", raw=raw)]
            return [WorkerEvent("error", text=msg, error=msg, raw=raw)]
        if t == "done":
            return [WorkerEvent("completed", text=str(e.get("text") or ""), raw=raw)]
        return []
    if etype == "run_result":
        return [WorkerEvent("completed", text=str(ev.get("text") or ""), raw=raw)]
    if etype == "run_aborted":
        return [WorkerEvent("warning", text=f"run aborted ({ev.get('reason', '?')}): {ev.get('message', '')}", raw=raw)]
    if etype == "error":
        return [WorkerEvent("error", text=str(ev.get("message") or ""), error=str(ev.get("message") or ""), raw=raw)]
    return []


def normalize_codex_event(ev: dict[str, Any]) -> list[WorkerEvent]:
    """One Codex `exec --json` line (already parsed) -> normalized events (the same vocabulary as Cline)."""
    etype = ev.get("type", "")
    item = ev.get("item") or {}
    itype = item.get("type", "")
    raw = redact(ev)
    if etype == "thread.started":
        return [WorkerEvent("session_started", session_id=str(ev.get("thread_id") or ""), raw=raw)]
    if itype in ("command_execution", "file_change", "mcp_tool_call", "web_search"):
        name = itype if itype != "mcp_tool_call" else f"{item.get('server', '?')}.{item.get('tool', '?')}"
        files = [(str(c.get("kind", "")), str(c["path"])) for c in item.get("changes") or []
                 if isinstance(c, dict) and c.get("path")]
        if etype == "item.started":
            return [WorkerEvent("tool_started", text=_text(item.get("command") or item.get("arguments"), 400),
                                tool_name=name, tool_id=str(item.get("id", "")), files=files, raw=raw)]
        if etype == "item.completed":
            return [WorkerEvent("tool_finished", text=_text(item.get("aggregated_output"), 1500), tool_name=name,
                                tool_id=str(item.get("id", "")), files=files, raw=raw)]
    if etype == "item.completed" and itype == "agent_message":
        return [WorkerEvent("assistant_text", text=str(item.get("text", "")), raw=raw)]
    if etype == "item.completed" and itype == "reasoning" and item.get("text"):
        return [WorkerEvent("reasoning", text=str(item["text"]), raw=raw)]
    if etype == "item.completed" and itype == "error":
        return [WorkerEvent("error", text=str(item.get("message", "")), error=str(item.get("message", "")), raw=raw)]
    if etype == "turn.completed":
        return [WorkerEvent("usage", raw=raw), WorkerEvent("completed", raw=raw)]
    if etype in ("turn.failed", "error"):
        msg = (ev.get("error") or {}).get("message") if etype == "turn.failed" else ev.get("message")
        return [WorkerEvent("error", text=str(msg), error=str(msg), raw=raw)]
    return []
