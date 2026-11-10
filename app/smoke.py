"""Command-line smoke tests for the two real adapters (same code path as the GUI).

    python -m app.smoke claude|worker|all [worker-profile-id]

On success the verified executable is recorded in settings, which is the gate
for using the real adapters in runs. The worker test runs the given V5 worker
profile (default: the default profile - the legacy NInfer worker on a V4 setup).
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile

from app.config.discovery import autofill
from app.config.settings import SettingsStore
from app.orchestration.orchestrator import Engine


def main(argv: list[str]) -> int:
    which = argv[1] if len(argv) > 1 else "all"
    profile_id = argv[2] if len(argv) > 2 else None
    store = SettingsStore()
    settings = store.load()
    autofill(settings)

    def emit(kind, payload):
        if kind == "log":
            print(f"[{payload['source']:<6}] {payload['text']}", flush=True)

    async def run() -> list[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(settings, emit, db_path=f"{tmp}/smoke.db")
            results = []
            if which in ("claude", "all"):
                results.append(await engine.smoke_test_claude())
            if which in ("worker", "all"):
                results.append(await engine.smoke_test_worker(profile_id))
            settings.claude.verified_executable = engine.settings.claude.verified_executable
            settings.claude.verified_at = engine.settings.claude.verified_at
            settings.worker.verified_executable = engine.settings.worker.verified_executable
            settings.worker.verified_codex_home = engine.settings.worker.verified_codex_home
            settings.worker.verified_at = engine.settings.worker.verified_at
            settings.worker_profiles.profiles = engine.settings.worker_profiles.profiles  # per-profile verification
            engine.db.close()
            return results

    results = asyncio.run(run())
    store.save(settings)
    print("\n=== RESULTS ===")
    for r in results:
        print(json.dumps({k: r[k] for k in ("target", "ok", "executable", "detail")}, indent=1))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
