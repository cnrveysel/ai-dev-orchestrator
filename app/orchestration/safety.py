"""Classification of shell commands that need explicit user approval."""
from __future__ import annotations

import re

_DESTRUCTIVE = [
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard"),
    (r"\bgit\s+clean\b", "git clean"),
    (r"\bgit\s+push\b", "git push"),
    (r"\bgit\s+(rebase|filter-branch|filter-repo)\b", "history rewrite"),
    (r"\bgit\s+(checkout|restore)\s+(--\s+)?\.(\s|$)", "discard working tree changes"),
    (r"\bgit\s+branch\s+-D\b", "force-delete branch"),
    (r"\bgit\s+stash\s+(drop|clear)\b", "drop stash"),
    (r"\brm\s+(-\w*[rf]\w*\s+)", "rm -r/-f"),
    (r"\b(rmdir|rd)\s+/s\b", "rmdir /s"),
    (r"\bdel\s+(/\w\s+)*/[sq]\b", "del /s or /q"),
    (r"\b(del|erase)\b", "delete files"),
    (r"\bRemove-Item\b", "Remove-Item"),
    (r"\bformat\s+[a-z]:", "format drive"),
    (r"\b(shutdown|reg\s+delete|diskpart|cipher\s+/w)\b", "system command"),
    (r"\bsetx\b", "modify persistent environment variables"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in _DESTRUCTIVE]


def destructive_reason(command: str) -> str | None:
    """Return a human-readable reason if the command needs approval, else None."""
    for rx, label in _COMPILED:
        if rx.search(command):
            return label
    return None
