"""Settings > Worker Profiles (V5).

Deliberately generic: a profile is its identity, worker runtime (Codex / Cline), executable, the runtime's own
configuration (Codex: CODEX_HOME; Cline: provider + model + optional configuration dir), process-local environment
and some display metadata. Provider/model options are NOT re-implemented as widgets - "Open config.toml" edits the profile's
own Codex configuration, which stays authoritative. Changes apply to NEW runs only; a run keeps the snapshot of the
profile it started with, also on resume. Quick checks never launch anything; the full test (a real, tiny worker
call) runs only when asked for.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from app.config import codex_config
from app.config.discovery import discover_cline_executables, discover_codex_builds
from app.config.settings import Settings
from app.workers.profile_registry import ProfileError, WorkerProfileRegistry
from app.workers.profiles import LEGACY_PROFILE_ID, MODE_MANAGED, RUNTIME_CLINE, RUNTIME_CODEX, WorkerProfile, redact_env
from app.workers.runtimes import WORKER_RUNTIMES, cline_providers, get_runtime, probe_sync

InUseProvider = Callable[[], "dict[str, list[int]] | None"]


def _env_text(env: dict[str, str]) -> str:
    return "\n".join(f"{k}={v}" for k, v in env.items())


def _parse_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise ProfileError(f"invalid env line (expected KEY=VALUE): {line}")
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def _parse_names(text: str) -> list[str]:
    return [n.strip() for n in text.replace(",", "\n").splitlines() if n.strip()]


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class ProfileEditDialog(QDialog):
    """Add / Register existing (Codex) / Edit. The form starts with the WORKER RUNTIME and shows only that runtime's
    fields: Codex = executable + CODEX_HOME (+ config.toml handling); Cline = the native cline.exe, the provider and
    model it runs (-P / -m; Cline keeps the credentials) and an optional Cline configuration directory. CODEX_HOME is
    never shown for Cline. The runtime is fixed once a profile exists. Produces `self.values` on accept."""

    def __init__(self, kind: str, registry: WorkerProfileRegistry, profile: WorkerProfile | None = None, parent=None,
                 runtime: str = RUNTIME_CODEX):
        super().__init__(parent)
        self.kind, self.registry, self.values = kind, registry, {}
        titles = {"create": "Add worker profile", "register": "Register an existing CODEX_HOME (Codex)",
                  "edit": "Edit worker profile"}
        self.setWindowTitle(titles[kind])
        self.resize(720, 620)
        p = profile or WorkerProfile(id="", runtime=runtime)
        f = QFormLayout()
        self.form = f
        self.runtime = QComboBox()
        for rid, r in WORKER_RUNTIMES.items():
            self.runtime.addItem(r.display_name, rid)
        self.runtime.setCurrentIndex(max(0, self.runtime.findData(p.runtime if kind != "register" else RUNTIME_CODEX)))
        self.runtime.setEnabled(kind == "create")  # another runtime = another (explicit) profile
        f.addRow("Worker runtime:", self.runtime)
        self.name = QLineEdit(p.display_name)
        f.addRow("Display name:", self.name)
        self.pid = QLineEdit(p.id)
        self.pid.setPlaceholderText("automatic (from the name)")
        self.pid.setEnabled(kind == "register")
        f.addRow("Profile id (fixed once created):", self.pid)
        # ---- Codex
        self._codex_rows: list[QWidget] = []
        self.exe = QComboBox()
        self.exe.setEditable(True)
        builds = [c.path for c in discover_codex_builds()]
        self.exe.addItems(builds)
        self.exe.setCurrentText(p.codex_executable or (builds[0] if builds else ""))
        row = self._with_browse(self.exe, file=True)
        f.addRow("Codex executable:", row)
        self._codex_rows.append(row)
        self.home = QComboBox()
        self.home.setEditable(True)
        self.template = QComboBox()
        self.new_cfg: dict[str, QWidget] = {}
        if kind == "register":
            known = {os.path.normcase(x.codex_home) for x in registry.profiles() if x.codex_home}
            self.home.addItems([h for h in codex_config.discover_codex_homes(managed_root=registry.managed_root())
                                if os.path.normcase(h) not in known])
            row = self._with_browse(self.home, file=False)
            f.addRow("Existing CODEX_HOME:", row)
            note = QLabel("External profile: AI Studio references this folder and never rewrites its config.toml.")
            f.addRow(note)
            self._codex_rows += [row, note]
        elif kind == "edit":
            self.home.setCurrentText(p.codex_home)
            row = self._with_browse(self.home, file=False)
            f.addRow("CODEX_HOME:", row)
            self._codex_rows.append(row)
        else:
            self.template.addItem("(new minimal config.toml)", "")
            for x in registry.profiles():
                if x.runtime == RUNTIME_CODEX and codex_config.config_path(x.codex_home).is_file():
                    self.template.addItem(f"copy config.toml of {x.name}", x.codex_home)
            f.addRow("Configuration:", self.template)
            box = QWidget()
            g = QFormLayout(box)
            g.setContentsMargins(0, 0, 0, 0)
            for key, label, ph in [("provider_id", "Provider id:", "e.g. ninfer_swift"),
                                   ("provider_name", "Provider name:", "e.g. NInfer Swift"),
                                   ("base_url", "Base URL:", "http://host:port/v1"),
                                   ("model", "Model:", "model name served by the provider"),
                                   ("api_key_env", "API key env var NAME:", "optional, e.g. OPENROUTER_API_KEY")]:
                w = QLineEdit()
                w.setPlaceholderText(ph)
                self.new_cfg[key] = w
                g.addRow(label, w)
            ctx = QSpinBox()
            ctx.setRange(0, 10_000_000)
            self.new_cfg["context_window"] = ctx
            g.addRow("Context window (0 = unset):", ctx)
            self.template.currentIndexChanged.connect(lambda _i: box.setEnabled(not self.template.currentData()))
            f.addRow(box)
            root = QLabel(f"The new CODEX_HOME is created under {registry.managed_root()}")
            root.setWordWrap(True)
            f.addRow(root)
            self._codex_rows += [self.template, box, root]
        # ---- Cline
        self._cline_rows: list[QWidget] = []
        self.cline_exe = QComboBox()
        self.cline_exe.setEditable(True)
        found = [c.path for c in discover_cline_executables()]
        self.cline_exe.addItems(found)
        self.cline_exe.setCurrentText(p.cline_executable or (found[0] if found else ""))
        row = self._with_browse(self.cline_exe, file=True)
        f.addRow("Cline executable (native cline.exe):", row)
        self._cline_rows.append(row)
        self.cline_config = QLineEdit(p.cline_config_dir)
        self.cline_config.setPlaceholderText("empty = Cline's default configuration (~/.cline)")
        f.addRow("Cline configuration dir (--config):", self.cline_config)
        self._cline_rows.append(self.cline_config)
        self._saved_models = cline_providers(p.cline_config_dir) or {}
        self.cline_provider = QComboBox()
        self.cline_provider.setEditable(True)
        self.cline_provider.addItems(sorted(self._saved_models))
        self.cline_provider.setCurrentText(p.provider if p.runtime == RUNTIME_CLINE else "")
        f.addRow("Provider (-P):", self.cline_provider)
        self._cline_rows.append(self.cline_provider)
        self.cline_model = QLineEdit(p.model if p.runtime == RUNTIME_CLINE else "")
        f.addRow("Model (-m, locked with every run):", self.cline_model)
        self._cline_rows.append(self.cline_model)
        self.cline_provider.currentTextChanged.connect(self._provider_changed)
        if not self.cline_model.text():
            self._provider_changed(self.cline_provider.currentText())
        cline_note = QLabel("Credentials stay in Cline's own configuration (`cline auth <provider>`). "
                            + " ".join(f"• {n}." for n in WORKER_RUNTIMES[RUNTIME_CLINE].notes))
        cline_note.setWordWrap(True)
        f.addRow(cline_note)
        self._cline_rows.append(cline_note)
        # ---- shared
        self.env = QPlainTextEdit(_env_text(p.extra_env))
        self.env.setMaximumHeight(60)
        self.env.setPlaceholderText("KEY=VALUE per line (non-secret, child process only), e.g. RUST_MIN_STACK=33554432")
        f.addRow("Extra env:", self.env)
        self.required = QLineEdit(", ".join(p.required_env))
        self.required.setPlaceholderText("names only, e.g. OPENROUTER_API_KEY - values are never stored")
        f.addRow("Required env variables:", self.required)
        self.pass_model = QCheckBox("Pass --model (otherwise config.toml decides the model)")
        self.pass_model.setChecked(p.pass_model_flag)
        f.addRow(self.pass_model)
        self.model = QLineEdit(p.model if p.runtime == RUNTIME_CODEX else "")
        f.addRow("Model (display / --model):", self.model)
        self.provider = QLineEdit(p.provider if p.runtime == RUNTIME_CODEX else "")
        f.addRow("Provider (display):", self.provider)
        self.base_url = QLineEdit(p.base_url)
        f.addRow("Base URL (display):", self.base_url)
        self._codex_rows += [self.pass_model, self.model, self.provider, self.base_url]
        self.args = QPlainTextEdit("\n".join(p.extra_args))
        self.args.setMaximumHeight(50)
        self.args.setPlaceholderText("one argument per line")
        f.addRow("Extra CLI args:", self.args)
        self.notes = QLineEdit(p.notes)
        f.addRow("Notes:", self.notes)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(f)
        lay.addWidget(buttons)
        self.runtime.currentIndexChanged.connect(lambda _i: self._runtime_changed())
        self._runtime_changed()

    def selected_runtime(self) -> str:
        return self.runtime.currentData() or RUNTIME_CODEX

    def _runtime_changed(self) -> None:
        cline = self.selected_runtime() == RUNTIME_CLINE
        for w in self._codex_rows:
            self.form.setRowVisible(w, not cline)
        for w in self._cline_rows:
            self.form.setRowVisible(w, cline)
        self.pid.setEnabled(self.kind == "register" or (self.kind == "create" and cline))

    def _provider_changed(self, provider: str) -> None:
        saved = self._saved_models.get(provider.strip(), "")
        self.cline_model.setPlaceholderText(f"Cline's saved model for {provider}: {saved}" if saved else "model id")
        if saved and (not self.cline_model.text() or self.cline_model.text() in self._saved_models.values()):
            self.cline_model.setText(saved)

    def _with_browse(self, combo: QComboBox, file: bool) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(combo, 1)
        b = QPushButton("Browse…")

        def browse() -> None:
            if file:
                path, _ = QFileDialog.getOpenFileName(w, "Select the executable", combo.currentText(),
                                                      "Executables (*.exe)")
            else:
                path = QFileDialog.getExistingDirectory(w, "Select CODEX_HOME", combo.currentText())
            if path:
                combo.setCurrentText(path.replace("/", "\\"))
        b.clicked.connect(browse)
        h.addWidget(b)
        return w

    def _ok(self) -> None:
        rt = self.selected_runtime()
        try:
            v: dict[str, Any] = {
                "runtime": rt, "display_name": self.name.text().strip(),
                "extra_env": _parse_env(self.env.toPlainText()), "required_env": _parse_names(self.required.text()),
                "extra_args": _lines(self.args.toPlainText()), "notes": self.notes.text().strip()}
        except ProfileError as exc:
            QMessageBox.warning(self, "Worker profile", str(exc))
            return
        if not v["display_name"]:
            QMessageBox.warning(self, "Worker profile", "Enter a display name.")
            return
        if rt == RUNTIME_CLINE:
            v.update(cline_executable=self.cline_exe.currentText().strip(),
                     cline_config_dir=self.cline_config.text().strip(),
                     provider=self.cline_provider.currentText().strip(), model=self.cline_model.text().strip(),
                     profile_id=self.pid.text().strip() if self.kind == "create" else "")
            if not v["provider"] or not v["model"]:
                QMessageBox.warning(self, "Worker profile", "A Cline profile needs a provider and a model.")
                return
            self.values = v
            self.accept()
            return
        v.update(codex_executable=self.exe.currentText().strip(), pass_model_flag=self.pass_model.isChecked(),
                 model=self.model.text().strip(), provider=self.provider.text().strip(),
                 base_url=self.base_url.text().strip())
        if self.kind in ("register", "edit"):
            v["codex_home"] = self.home.currentText().strip()
        if self.kind == "register":
            v["profile_id"] = self.pid.text().strip()
        if self.kind == "create":
            v["template_home"] = self.template.currentData() or ""
            v["new_config"] = {k: (w.value() if isinstance(w, QSpinBox) else w.text().strip())
                               for k, w in self.new_cfg.items()}
        self.values = v
        self.accept()


class WorkerProfilesTab(QWidget):
    def __init__(self, settings: Settings, in_use_provider: InUseProvider | None = None,
                 on_edit_legacy: Callable[[], None] | None = None,
                 on_full_test: Callable[[str], None] | None = None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.registry = WorkerProfileRegistry(settings)
        self.in_use_provider = in_use_provider
        self.on_edit_legacy = on_edit_legacy
        self.on_full_test = on_full_test
        self.default_combo = QComboBox()
        self.default_combo.currentIndexChanged.connect(self._default_changed)
        top = QHBoxLayout()
        top.addWidget(QLabel("Default worker profile (new runs):"))
        top.addWidget(self.default_combo, 1)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(lambda *_: self._show_details())
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.list)
        split.addWidget(self.details)
        split.setSizes([260, 460])
        self.buttons: dict[str, QPushButton] = {}
        grid = QGridLayout()
        specs = [("add", "Add…", self._add), ("register", "Register Existing…", self._register),
                 ("duplicate", "Duplicate…", self._duplicate), ("edit", "Edit…", self._edit),
                 ("toggle", "Disable", self._toggle), ("delete", "Delete", self._delete),
                 ("test", "Test (quick)", self._test), ("full_test", "Full test…", self._full_test),
                 ("default", "Set Default", self._set_default), ("open_home", "Open CODEX_HOME", self._open_home),
                 ("open_config", "Open config.toml", self._open_config)]
        for i, (key, text, slot) in enumerate(specs):
            b = QPushButton(text)
            b.clicked.connect(slot)
            self.buttons[key] = b
            grid.addWidget(b, i // 6, i % 6)
        self.buttons["register"].setToolTip("Codex: reference an existing CODEX_HOME (e.g. ~/.codex-lmstudio)")
        note = QLabel("Profile changes apply to NEW runs only: a run keeps the worker profile (and runtime) it started "
                      "with, also when it is resumed - nothing ever falls back to another profile or runtime. Codex: "
                      "config.toml is authoritative. Cline: credentials stay in Cline's configuration. Secret values "
                      "are never stored (list the NAMES of required environment variables).")
        note.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        lay.addLayout(grid)
        lay.addWidget(note)
        self.reload()

    # ------------------------------------------------------------ view

    def selected(self) -> WorkerProfile | None:
        it = self.list.currentItem()
        return self.registry.get(it.data(Qt.UserRole)) if it else None

    def reload(self, select: str | None = None) -> None:
        keep = select or (self.list.currentItem().data(Qt.UserRole) if self.list.currentItem() else None)
        default = self.registry.default_id()
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.registry.profiles():
            tags = [t for t, on in (("default", p.id == default), ("disabled", not p.enabled),
                                    ("managed", p.mode == MODE_MANAGED), ("verified", p.is_verified())) if on]
            runtime = get_runtime(p.runtime)
            it = QListWidgetItem(f"{p.name}\n    Runtime: {runtime.display_name if runtime else p.runtime}"
                                 + (f"  [{', '.join(tags)}]" if tags else ""))
            it.setData(Qt.UserRole, p.id)
            it.setToolTip(f"{p.id}\n{p.executable_path}\n"
                          + (p.codex_home if p.runtime == RUNTIME_CODEX else f"-P {p.provider} -m {p.model}"))
            self.list.addItem(it)
            if p.id == (keep or default):
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)
        self.default_combo.blockSignals(True)
        self.default_combo.clear()
        for p in self.registry.profiles():
            if p.enabled:
                self.default_combo.addItem(p.name, p.id)
        self.default_combo.setCurrentIndex(max(0, self.default_combo.findData(default)))
        self.default_combo.blockSignals(False)
        self._show_details()

    def _show_details(self, extra: str = "") -> None:
        p = self.selected()
        for key, b in self.buttons.items():
            b.setEnabled(p is not None or key in ("add", "register"))
        if p is None:
            self.details.setPlainText("")
            return
        legacy = p.id == LEGACY_PROFILE_ID
        self.buttons["toggle"].setText("Enable" if not p.enabled else "Disable")
        self.buttons["toggle"].setEnabled(not legacy)
        self.buttons["delete"].setEnabled(not legacy)
        codex = p.runtime == RUNTIME_CODEX
        self.buttons["open_home"].setEnabled(codex)
        self.buttons["open_config"].setEnabled(codex)
        if not codex:
            self.details.setPlainText("\n".join(self._runtime_details(p) + ([f"Notes: {p.notes}"] if p.notes else [])
                                                + (["", extra] if extra else [])))
            return
        info = codex_config.inspect_home(p.codex_home)
        cfg = (f"provider {info.provider or '?'}, model {info.model or '?'}, base URL {info.base_url or '?'}"
               + (f", context {info.context_window}" if info.context_window else "")) if info.has_config else \
            ("could not be parsed: " + info.parse_error if info.parse_error else "missing")
        env = ", ".join(f"{k}={v}" for k, v in redact_env(p.extra_env).items()) or "(none)"
        verified = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.verified_at)) if p.is_verified() else "no"
        lines = [f"{p.name}   ({p.id}, revision {p.revision()})",
                 "Mode: " + ("managed - created by AI Studio" if p.mode == MODE_MANAGED else
                             "external - AI Studio never rewrites its config.toml")
                 + ("  |  the V4 worker settings (edit them on the Worker tab)" if legacy else ""),
                 f"Enabled: {'yes' if p.enabled else 'NO'}    Default: {'yes' if p.id == self.registry.default_id() else 'no'}",
                 "Runtime: Codex",
                 f"Executable: {p.codex_executable or '(not set)'}",
                 f"CODEX_HOME: {p.codex_home or '(not set)'}",
                 f"config.toml: {cfg}",
                 f"Model: {p.model or '(config.toml)'}" + ("  (passed as --model)" if p.pass_model_flag and p.model
                                                          else "  (display only)" if p.model else ""),
                 f"Provider: {p.provider or '(config.toml)'}    Base URL: {p.base_url or '(config.toml)'}",
                 f"Extra env (process-local): {env}",
                 f"Required env (names): {', '.join(p.required_env) or '(none)'}",
                 f"Extra args: {' '.join(p.extra_args) or '(none)'}",
                 f"Verified by a full test: {verified}"]
        if p.notes:
            lines.append(f"Notes: {p.notes}")
        if extra:
            lines += ["", extra]
        self.details.setPlainText("\n".join(lines))

    def _runtime_details(self, p: WorkerProfile) -> list[str]:
        """Details of a non-Codex profile (Cline)."""
        runtime = get_runtime(p.runtime)
        env = ", ".join(f"{k}={v}" for k, v in redact_env(p.extra_env).items()) or "(none)"
        verified = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.verified_at)) if p.is_verified() else "no"
        saved = cline_providers(p.cline_config_dir)
        lines = [f"{p.name}   ({p.id}, revision {p.revision()})",
                 f"Enabled: {'yes' if p.enabled else 'NO'}    Default: {'yes' if p.id == self.registry.default_id() else 'no'}",
                 f"Runtime: {runtime.display_name if runtime else p.runtime}",
                 f"Executable: {p.cline_executable or '(not set)'}",
                 f"Provider (-P): {p.provider or '(not set)'}    Model (-m): {p.model or '(not set)'}",
                 f"Cline configuration: {p.cline_config_dir or '(Cline default, ~/.cline)'}",
                 "Cline providers configured: " + (", ".join(f"{k} ({v})" for k, v in sorted(saved.items()))
                                                   if saved else "(provider settings not found)"),
                 f"Extra env (process-local): {env}",
                 f"Required env (names): {', '.join(p.required_env) or '(none)'}",
                 f"Extra args: {' '.join(p.extra_args) or '(none)'}",
                 f"Verified by a full test: {verified}", "Runtime limits:"]
        lines += [f"  - {n}" for n in (runtime.notes if runtime else ())]
        return lines

    def _fail(self, exc: Exception) -> None:
        QMessageBox.warning(self, "Worker profiles", str(exc))

    # ------------------------------------------------------------ actions

    def _default_changed(self, _i: int) -> None:
        pid = self.default_combo.currentData()
        if pid and pid != self.registry.default_id():
            try:
                self.registry.set_default(pid)
            except ProfileError as exc:
                self._fail(exc)
            self.reload()

    def _add(self) -> None:
        dlg = ProfileEditDialog("create", self.registry, parent=self)
        if not dlg.exec():
            return
        v = dlg.values
        if v["runtime"] == RUNTIME_CLINE:
            try:
                p = self.registry.create_cline(v["display_name"], v["cline_executable"], v["provider"], v["model"],
                                               profile_id=v["profile_id"], cline_config_dir=v["cline_config_dir"],
                                               extra_env=v["extra_env"], required_env=v["required_env"],
                                               extra_args=v["extra_args"], notes=v["notes"])
            except ProfileError as exc:
                self._fail(exc)
                return
            self.reload(p.id)
            return
        try:
            text = ""
            if not v["template_home"]:
                n = v["new_config"]
                if not (n["provider_id"] and n["base_url"] and n["model"]):
                    raise ProfileError("a new config.toml needs a provider id, a base URL and a model "
                                       "(or copy the config.toml of an existing profile)")
                text = codex_config.minimal_config(n["model"], n["provider_id"], n["provider_name"], n["base_url"],
                                                   context_window=n["context_window"], api_key_env=n["api_key_env"])
                if n["api_key_env"] and n["api_key_env"] not in v["required_env"]:
                    v["required_env"].append(n["api_key_env"])
            p = self.registry.create_managed(v["display_name"], v["codex_executable"], template_home=v["template_home"],
                                             config_text=text, extra_env=v["extra_env"],
                                             required_env=v["required_env"], notes=v["notes"])
            self.registry.update(p.id, pass_model_flag=v["pass_model_flag"], model=v["model"] or p.model,
                                 extra_args=v["extra_args"], provider=v["provider"] or p.provider,
                                 base_url=v["base_url"] or p.base_url)
        except (ProfileError, codex_config.CodexConfigError) as exc:
            self._fail(exc)
            return
        self.reload(p.id)

    def _register(self) -> None:
        dlg = ProfileEditDialog("register", self.registry, parent=self)
        if not dlg.exec():
            return
        v = dlg.values
        try:
            p = self.registry.register_existing(v["codex_home"], v["codex_executable"], v["display_name"],
                                                profile_id=v["profile_id"], extra_env=v["extra_env"],
                                                required_env=v["required_env"], notes=v["notes"])
            self.registry.update(p.id, pass_model_flag=v["pass_model_flag"], model=v["model"] or p.model,
                                 extra_args=v["extra_args"], provider=v["provider"] or p.provider,
                                 base_url=v["base_url"] or p.base_url)
        except ProfileError as exc:
            self._fail(exc)
            return
        self.reload(p.id)

    def _duplicate(self) -> None:
        src = self.selected()
        if src is None:
            return
        name, ok = QInputDialog.getText(self, "Duplicate worker profile", "Name of the copy:", text=f"{src.name} (copy)")
        if not ok or not name.strip():
            return
        if src.runtime != RUNTIME_CODEX:
            try:
                p = self.registry.duplicate(src.id, name)
            except ProfileError as exc:
                self._fail(exc)
                return
            self.reload(p.id)
            return
        r = QMessageBox.question(self, "Duplicate worker profile",
                                 "Create a NEW managed CODEX_HOME with a copy of this profile's config.toml?\n\n"
                                 "Yes: separate home (sessions and config are not shared).\n"
                                 "No: the copy references the same CODEX_HOME.",
                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return
        try:
            p = self.registry.duplicate(src.id, name, copy_home=r == QMessageBox.Yes)
        except ProfileError as exc:
            self._fail(exc)
            return
        self.reload(p.id)

    def _edit(self) -> None:
        p = self.selected()
        if p is None:
            return
        if p.id == LEGACY_PROFILE_ID:
            QMessageBox.information(self, "Worker profiles", "The legacy profile is the V4 worker configuration: "
                                                             "edit it on the Worker tab.")
            if self.on_edit_legacy:
                self.on_edit_legacy()
            return
        dlg = ProfileEditDialog("edit", self.registry, p, parent=self)
        if not dlg.exec():
            return
        v = dlg.values
        keys = (("display_name", "cline_executable", "cline_config_dir", "provider", "model", "extra_env",
                 "required_env", "extra_args", "notes") if p.runtime == RUNTIME_CLINE else
                ("display_name", "codex_executable", "codex_home", "extra_env", "required_env", "pass_model_flag",
                 "model", "provider", "base_url", "extra_args", "notes"))
        try:
            self.registry.update(p.id, **{k: v[k] for k in keys})
        except ProfileError as exc:
            self._fail(exc)
            return
        self.reload(p.id)

    def _toggle(self) -> None:
        p = self.selected()
        if p is None:
            return
        try:
            self.registry.set_enabled(p.id, not p.enabled)
        except ProfileError as exc:
            self._fail(exc)
        self.reload(p.id)

    def _delete(self) -> None:
        p = self.selected()
        if p is None:
            return
        in_use = self.in_use_provider() if self.in_use_provider else {}
        if in_use is None:
            self._fail(ProfileError("could not check which runs use this profile; nothing was deleted"))
            return
        if p.id in in_use:
            runs = ", ".join(f"#{i}" for i in in_use[p.id])
            self._fail(ProfileError(f"{p.name} is used by run(s) {runs} that can still be resumed. Finish them "
                                    "first, or disable the profile instead."))
            return
        kept = (f"Its CODEX_HOME folder is NOT deleted:\n{p.codex_home}" if p.runtime == RUNTIME_CODEX
                else "Cline's own configuration and sessions are NOT changed.")
        if QMessageBox.question(self, "Delete worker profile",
                                f"Remove the profile '{p.name}' from AI Studio?\n\n{kept}") != QMessageBox.Yes:
            return
        try:
            self.registry.delete(p.id, in_use)
        except ProfileError as exc:
            self._fail(exc)
        self.reload()

    def _test(self) -> None:
        p = self.selected()
        if p is None:
            return
        rep = self.registry.validate(p.id)
        text = rep.summary()
        if rep.info.get("env_refs"):
            text += "\nconfig.toml reads: " + ", ".join(rep.info["env_refs"])
        if p.runtime != RUNTIME_CODEX and not rep.errors:  # "the CLI responds" (a --version call, no model call)
            try:
                text += f"\nCLI responds: {probe_sync(p) or '(no probe)'}"
            except Exception as exc:
                rep.ok = False
                text += f"\nerror: the CLI does not respond: {exc}"
        self._show_details("QUICK CHECK:\n" + text)
        (QMessageBox.information if rep.ok else QMessageBox.warning)(self, "Worker profile quick check", text)

    def _full_test(self) -> None:
        p = self.selected()
        if p is not None and self.on_full_test:
            self.on_full_test(p.id)

    def _set_default(self) -> None:
        p = self.selected()
        if p is None:
            return
        try:
            self.registry.set_default(p.id)
        except ProfileError as exc:
            self._fail(exc)
        self.reload(p.id)

    def _open_home(self) -> None:
        p = self.selected()
        if p is not None and p.codex_home and os.path.isdir(os.path.expandvars(p.codex_home)):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.expandvars(p.codex_home)))

    def _open_config(self) -> None:
        p = self.selected()
        if p is None:
            return
        path = str(codex_config.config_path(p.codex_home))
        if not os.path.isfile(path):
            self._fail(ProfileError(f"{path} does not exist"))
            return
        if p.mode != MODE_MANAGED:
            QMessageBox.information(self, "config.toml", "This is an external profile: AI Studio never rewrites its "
                                                         "config.toml. Edit it only while no run uses the profile.")
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            subprocess.Popen(["notepad.exe", path])
