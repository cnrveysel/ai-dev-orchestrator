"""Headless GUI tests of Settings > Worker Profiles (V5). No worker is launched."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from app.config.settings import Settings, SettingsStore  # noqa: E402
from app.gui.settings_dialog import SettingsDialog  # noqa: E402
from app.workers.profile_registry import WorkerProfileRegistry  # noqa: E402
from app.workers.profiles import LEGACY_PROFILE_ID  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    exe = tmp_path / "bin" / "codex-lmstudio.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"MZ")
    home = tmp_path / ".codex-lmstudio"
    home.mkdir()
    (home / "config.toml").write_text('model = "Qwen3.8 27B"\nmodel_provider = "lmstudio_lan"\n', encoding="utf-8")
    s = Settings()
    s.integrations.metrics_enabled = False
    reg = WorkerProfileRegistry(s)
    p = reg.register_existing(str(home), str(exe), "LM Studio Qwen")
    shown = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, lambda *a, _n=name, **k: shown.append((_n, a[1:])))
    return s, p, shown


def items(tab):
    return [tab.list.item(i).text() for i in range(tab.list.count())]


def test_profiles_tab_lists_default_and_registered_profiles(qapp, setup):
    s, p, shown = setup
    dlg = SettingsDialog(s)
    tab = dlg.profiles_tab
    assert items(tab)[0].startswith("NInfer Qwen (legacy settings)") and "[default]" in items(tab)[0]
    assert items(tab)[1].startswith("LM Studio Qwen")
    assert [tab.default_combo.itemData(i) for i in range(tab.default_combo.count())] == [LEGACY_PROFILE_ID, p.id]
    tab.list.setCurrentRow(1)
    text = tab.details.toPlainText()
    assert "external - AI Studio never rewrites its config.toml" in text and "lmstudio_lan" in text
    assert tab.buttons["delete"].isEnabled() and tab.buttons["toggle"].isEnabled()
    tab.list.setCurrentRow(0)
    assert not tab.buttons["delete"].isEnabled() and not tab.buttons["toggle"].isEnabled()
    tab.list.setCurrentRow(1)
    tab.buttons["test"].click()  # quick check only: nothing is launched
    assert shown[-1][0] == "information" and ": OK" in shown[-1][1][1] and "not verified yet" in shown[-1][1][1]
    assert "QUICK CHECK" in tab.details.toPlainText()


def test_delete_is_refused_while_a_run_uses_the_profile(qapp, setup, monkeypatch):
    s, p, shown = setup
    dlg = SettingsDialog(s, in_use_provider=lambda: {p.id: [7]})
    tab = dlg.profiles_tab
    tab.list.setCurrentRow(1)
    tab.buttons["delete"].click()
    assert any("#7" in str(args) for _n, args in shown)
    assert tab.registry.get(p.id) is not None
    dlg2 = SettingsDialog(s, in_use_provider=lambda: None)  # unknown -> never delete
    dlg2.profiles_tab.list.setCurrentRow(1)
    dlg2.profiles_tab.buttons["delete"].click()
    assert dlg2.profiles_tab.registry.get(p.id) is not None


def test_real_mode_requires_a_verified_default_profile(qapp, setup):
    s, p, shown = setup
    s.orchestration.worker_mode = "ninfer"
    reg = WorkerProfileRegistry(s)
    reg.record_verification(LEGACY_PROFILE_ID, True, s.worker.executable, s.worker.codex_home)
    s.worker.executable = ""  # keep the legacy tab's executable check out of the way
    s.orchestration.worker_mode = "mock"
    dlg = SettingsDialog(s)
    dlg.profiles_tab.registry.set_default(p.id)
    dlg.o_worker.setCurrentText("ninfer")
    dlg._save()
    assert not dlg._accepted and shown[-1][0] == "critical" and "verified default worker profile" in shown[-1][1][1]
    dlg.profiles_tab.registry.record_verification(p.id, True, p.codex_executable, p.codex_home)
    dlg._save()
    assert dlg._accepted and dlg.settings.worker_profiles.default_profile_id == p.id


def test_smoke_result_is_recorded_on_the_tested_profile(qapp, setup, tmp_path, monkeypatch):
    import threading
    import time
    from app.gui.bridge import EventBridge
    from app.gui.main_window import MainWindow
    from app.orchestration.orchestrator import EngineHost
    s, p, shown = setup
    store = SettingsStore(tmp_path / "settings.json")
    bridge = EventBridge()
    host = EngineHost(s, bridge.emit, str(tmp_path / "g.db"))
    win = MainWindow(s, store, bridge, host)
    host.start()
    payload = {"target": "worker", "ok": True, "executable": p.codex_executable, "detail": "ok",
               "extra": {"codex_home": p.codex_home, "reply": "QWEN_WORKER_OK", "exit_code": 0, "elapsed": 1.0,
                         "files_created": []},
               "verified_at": 5.0, "verified_executable": p.codex_executable, "verified_codex_home": p.codex_home,
               "profile_id": p.id, "profile_name": p.name}
    try:
        assert win._profiles_in_use() == {}
        threading.Thread(target=lambda: bridge.emit("smoke", payload)).start()
        t0 = time.time()
        while not shown and time.time() - t0 < 20:
            qapp.processEvents()
            time.sleep(0.01)
        assert "LM Studio Qwen" in shown[-1][1][0] and "PASSED" in shown[-1][1][0]
        saved = WorkerProfileRegistry(store.load())
        assert saved.require(p.id).is_verified() and not saved.legacy().is_verified()
    finally:
        win.close()  # shuts the engine host down


# ------------------------------------------------------------ V5.1: Codex + Cline in the same tab

@pytest.fixture
def with_cline(setup, tmp_path):
    import json as _json
    s, p, shown = setup
    exe = tmp_path / "npm" / "cline.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"MZ")
    cfg = tmp_path / ".cline" / "data" / "settings"
    cfg.mkdir(parents=True)
    (cfg / "providers.json").write_text(_json.dumps({"providers": {
        "openai-compatible": {"settings": {"model": "mimo-v2.6-pro", "apiKey": "sk-x"}},
        "cline": {"settings": {"model": "stealth/pixel-canary"}}}}), encoding="utf-8")
    return s, p, shown, str(exe)


def test_profiles_tab_renders_codex_and_cline_profiles(qapp, with_cline):
    s, codex, shown, exe = with_cline
    WorkerProfileRegistry(s).create_cline("Cline MiMo", exe, "openai-compatible", "mimo-v2.6-pro", profile_id="cline-mimo")
    dlg = SettingsDialog(s)
    tab = dlg.profiles_tab
    texts = items(tab)
    assert "Runtime: Codex" in texts[0] and "Runtime: Codex" in texts[1] and "Runtime: Cline" in texts[2]
    tab.list.setCurrentRow(2)
    d = tab.details.toPlainText()
    assert "Runtime: Cline" in d and "Provider (-P): openai-compatible" in d and "Model (-m): mimo-v2.6-pro" in d
    assert "CODEX_HOME" not in d and "sk-x" not in d and "no command sandbox" in d
    assert not tab.buttons["open_home"].isEnabled() and not tab.buttons["open_config"].isEnabled()
    tab.list.setCurrentRow(1)
    assert tab.buttons["open_home"].isEnabled() and "CODEX_HOME" in tab.details.toPlainText()


def test_edit_dialog_shows_only_the_selected_runtimes_fields(qapp, with_cline):
    from app.gui.worker_profiles_tab import ProfileEditDialog
    s, codex, shown, exe = with_cline
    dlg = ProfileEditDialog("create", WorkerProfileRegistry(s))
    assert dlg.runtime.isEnabled() and dlg.selected_runtime() == "codex"
    assert dlg.form.isRowVisible(dlg.template) and not dlg.form.isRowVisible(dlg.cline_model)
    dlg.runtime.setCurrentIndex(dlg.runtime.findData("cline"))
    assert not dlg.form.isRowVisible(dlg.template) and not dlg.form.isRowVisible(dlg.pass_model)
    assert dlg.form.isRowVisible(dlg.cline_provider) and dlg.form.isRowVisible(dlg.cline_model)
    assert dlg.pid.isEnabled()
    dlg.name.setText("Cline Pixel")
    dlg.cline_exe.setCurrentText(exe)
    dlg.cline_provider.setCurrentText("cline")
    assert dlg.cline_model.text() == "stealth/pixel-canary", "prefilled with Cline's saved model"
    dlg._ok()
    assert dlg.values["runtime"] == "cline" and dlg.values["provider"] == "cline"
    assert "codex_home" not in dlg.values and "template_home" not in dlg.values
    edit = ProfileEditDialog("edit", WorkerProfileRegistry(s), codex)
    assert not edit.runtime.isEnabled(), "the runtime of an existing profile is fixed"


def test_add_cline_profile_and_quick_test_never_runs_the_paid_test(qapp, with_cline, monkeypatch):
    from app.gui import worker_profiles_tab as tabmod
    from app.workers.cline_cli import ClineWorker
    s, codex, shown, exe = with_cline

    def fake_exec(dlg):
        dlg.runtime.setCurrentIndex(dlg.runtime.findData("cline"))
        dlg.name.setText("Cline MiMo")
        dlg.cline_exe.setCurrentText(exe)
        dlg.cline_provider.setCurrentText("openai-compatible")
        dlg._ok()
        return bool(dlg.values)

    monkeypatch.setattr(tabmod.ProfileEditDialog, "exec", fake_exec)
    probes = []
    monkeypatch.setattr(tabmod, "probe_sync", lambda p: probes.append(p.id) or "3.0.65")

    async def paid(self, metrics_url=""):
        raise AssertionError("the paid full test ran from a quick check")

    monkeypatch.setattr(ClineWorker, "smoke_test", paid)
    dlg = SettingsDialog(s)
    tab = dlg.profiles_tab
    tab._add()
    p = tab.registry.require("cline-mimo")
    assert p.runtime == "cline" and p.model == "mimo-v2.6-pro" and p.codex_home == ""
    tab.list.setCurrentRow(2)
    tab.buttons["test"].click()
    assert probes == ["cline-mimo"] and "CLI responds: 3.0.65" in shown[-1][1][1]
