"""Read-only access to a Codex profile home (CODEX_HOME/config.toml) for V5 worker profiles.

config.toml is the authoritative source for a profile's provider and model; AI Studio only reads it to display
metadata and to find the environment variable NAMES a provider needs. The only write is the creation of a NEW
managed home (a fresh folder that no run can be using yet). An existing config.toml is never rewritten here.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_NAME = "config.toml"
# Files copied when a managed home is created from a template home (never sessions, logs, databases or secrets).
TEMPLATE_FILES = (CONFIG_NAME, "AGENTS.md")
_ENV_REF = re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)|\$\{?([A-Z_][A-Z0-9_]*)\}?|%([A-Za-z_][A-Za-z0-9_]*)%")


class CodexConfigError(ValueError):
    pass


@dataclass
class CodexConfigInfo:
    """Display metadata of one CODEX_HOME. Never contains secret values."""
    home: str
    exists: bool = False
    has_config: bool = False
    parse_error: str = ""
    model: str = ""
    provider: str = ""
    provider_name: str = ""
    base_url: str = ""
    wire_api: str = ""
    context_window: int = 0
    env_refs: list[str] = field(default_factory=list)  # environment variable NAMES the provider reads
    mcp_servers: list[str] = field(default_factory=list)


def config_path(home: str) -> Path:
    return Path(os.path.expandvars(home)) / CONFIG_NAME


def read_config(home: str) -> dict[str, Any]:
    p = config_path(home)
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as exc:
        raise CodexConfigError(f"{p} does not exist") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CodexConfigError(f"{p} could not be parsed: {exc}") from exc


def _env_names(value: Any) -> list[str]:
    out: list[str] = []
    for s in (value if isinstance(value, list) else [value]):
        if isinstance(s, str):
            for m in _ENV_REF.finditer(s):
                name = next(g for g in m.groups() if g)
                if name not in out:
                    out.append(name)
    return out


def inspect_home(home: str) -> CodexConfigInfo:
    info = CodexConfigInfo(home=home)
    root = Path(os.path.expandvars(home)) if home else None
    info.exists = bool(root and root.is_dir())
    if not info.exists:
        return info
    info.has_config = config_path(home).is_file()
    if not info.has_config:
        return info
    try:
        cfg = read_config(home)
    except CodexConfigError as exc:
        info.parse_error = str(exc)
        return info
    info.model = str(cfg.get("model") or "")
    info.provider = str(cfg.get("model_provider") or "")
    try:
        info.context_window = int(cfg.get("model_context_window") or 0)
    except (TypeError, ValueError):
        info.context_window = 0
    prov = (cfg.get("model_providers") or {}).get(info.provider) or {}
    if isinstance(prov, dict):
        info.provider_name = str(prov.get("name") or "")
        info.base_url = str(prov.get("base_url") or "")
        info.wire_api = str(prov.get("wire_api") or "")
        refs: list[str] = []
        if isinstance(prov.get("env_key"), str) and prov["env_key"]:
            refs.append(prov["env_key"])
        auth = prov.get("auth")
        if isinstance(auth, dict):
            for key in ("command", "args"):
                refs += [n for n in _env_names(auth.get(key)) if n not in refs]
        info.env_refs = refs
    info.mcp_servers = sorted((cfg.get("mcp_servers") or {}).keys())
    return info


def discover_codex_homes(user_home: str | None = None, managed_root: str = "") -> list[str]:
    """Existing profile homes: ~/.codex-* folders and <managed root>/* folders that contain a config.toml. The
    normal ~/.codex is never offered."""
    base = Path(user_home or os.environ.get("USERPROFILE", str(Path.home())))
    found: list[str] = []
    candidates = sorted(p for p in base.glob(".codex-*") if p.is_dir() and p.name != ".codex-profiles")
    root = Path(managed_root) if managed_root else base / ".codex-profiles"
    if root.is_dir():
        candidates += sorted(p for p in root.iterdir() if p.is_dir())
    for p in candidates:
        if (p / CONFIG_NAME).is_file():
            found.append(str(p))
    return found


def default_managed_root() -> str:
    return str(Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".codex-profiles")


def _toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # a JSON string is a valid TOML basic string


def minimal_config(model: str, provider_id: str, provider_name: str, base_url: str, *, context_window: int = 0,
                   api_key_env: str = "") -> str:
    """A minimal config.toml using only keys the existing profiles on this machine already use. An API key is
    referenced by environment variable NAME through the same auth-command form the OpenRouter profile uses."""
    pid = re.sub(r"[^a-z0-9_]", "_", provider_id.lower()) or "provider"
    lines = ["# Created by AI Studio (managed worker profile). config.toml is the authoritative worker",
             "# configuration; edit it freely while no run uses this profile.",
             f"model = {_toml_str(model)}", f"model_provider = {_toml_str(pid)}"]
    if context_window:
        lines.append(f"model_context_window = {int(context_window)}")
    lines += ["", f"[model_providers.{pid}]", f"name = {_toml_str(provider_name or pid)}",
              f"base_url = {_toml_str(base_url)}", 'wire_api = "responses"']
    if api_key_env:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", api_key_env):
            raise CodexConfigError(f"invalid environment variable name: {api_key_env!r}")
        lines += ["", f"[model_providers.{pid}.auth]", 'command = "powershell"',
                  f"args = [\"-NoProfile\", \"-Command\", {_toml_str('Write-Output $env:' + api_key_env)}]"]
    return "\n".join(lines) + "\n"


def create_managed_home(home: str, *, template_home: str = "", config_text: str = "") -> list[str]:
    """Create a NEW managed CODEX_HOME. Refuses an existing non-empty folder (never overwrites a profile). Returns
    the files written."""
    root = Path(os.path.expandvars(home))
    if root.exists() and any(root.iterdir()):
        raise CodexConfigError(f"{root} already exists and is not empty; register it as an existing profile instead")
    if not template_home and not config_text:
        raise CodexConfigError("a new managed profile needs a template CODEX_HOME or a config.toml")
    if template_home and not config_path(template_home).is_file():
        raise CodexConfigError(f"template profile has no config.toml: {template_home}")
    if template_home:
        read_config(template_home)  # refuse to copy a config that does not parse
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if template_home:
        src = Path(os.path.expandvars(template_home))
        for name in TEMPLATE_FILES:
            if (src / name).is_file():
                shutil.copy2(src / name, root / name)
                written.append(name)
    else:
        tomllib.loads(config_text)  # must parse before it is written
        (root / CONFIG_NAME).write_text(config_text, encoding="utf-8")
        written.append(CONFIG_NAME)
    return written
