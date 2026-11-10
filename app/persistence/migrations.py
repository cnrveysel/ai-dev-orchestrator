"""Ordered schema migrations. Append new entries; never edit applied ones."""
from __future__ import annotations

import sqlite3

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal TEXT NOT NULL,
        repo_path TEXT NOT NULL,
        state TEXT NOT NULL,
        resume_state TEXT,
        status_message TEXT DEFAULT '',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        started_at REAL,
        finished_at REAL,
        interrupted INTEGER DEFAULT 0,
        git_head TEXT,
        git_branch TEXT,
        working_branch TEXT,
        initial_dirty_json TEXT DEFAULT '[]',
        current_task_id TEXT,
        plan_json TEXT,
        settings_snapshot_json TEXT DEFAULT '{}',
        orchestrator_mode TEXT,
        worker_mode TEXT
    );
    CREATE TABLE tasks (
        run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        task_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER DEFAULT 0,
        data_json TEXT NOT NULL,
        PRIMARY KEY (run_id, task_id)
    );
    CREATE TABLE task_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        task_id TEXT NOT NULL,
        number INTEGER NOT NULL,
        kind TEXT,
        started_at REAL,
        finished_at REAL,
        exit_code INTEGER,
        interrupted INTEGER DEFAULT 0,
        data_json TEXT NOT NULL
    );
    CREATE INDEX idx_attempts_task ON task_attempts(run_id, task_id, number);
    CREATE TABLE orchestrator_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        task_id TEXT,
        kind TEXT,
        executable TEXT,
        model TEXT,
        argv_json TEXT,
        started_at REAL,
        duration REAL,
        exit_code INTEGER,
        cost_usd REAL,
        usage_json TEXT,
        session_id TEXT,
        error TEXT
    );
    CREATE TABLE worker_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        attempt_id INTEGER,
        executable TEXT,
        argv_json TEXT,
        env_overrides_json TEXT,
        started_at REAL,
        duration REAL,
        exit_code INTEGER,
        timed_out INTEGER,
        cancelled INTEGER,
        tool_calls INTEGER,
        usage_json TEXT
    );
    CREATE TABLE logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        ts REAL NOT NULL,
        source TEXT NOT NULL,
        message TEXT NOT NULL
    );
    CREATE INDEX idx_logs_run ON logs(run_id, id);
    CREATE TABLE changed_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id INTEGER NOT NULL REFERENCES task_attempts(id) ON DELETE CASCADE,
        path TEXT NOT NULL,
        change TEXT,
        attributed_to_worker INTEGER NOT NULL,
        note TEXT
    );
    CREATE TABLE verification_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id INTEGER NOT NULL REFERENCES task_attempts(id) ON DELETE CASCADE,
        command TEXT,
        exit_code INTEGER,
        passed INTEGER,
        duration REAL,
        skipped_reason TEXT,
        output_tail TEXT
    );
    CREATE TABLE reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        task_id TEXT,
        attempt_id INTEGER REFERENCES task_attempts(id) ON DELETE CASCADE,
        decision TEXT,
        summary TEXT,
        data_json TEXT,
        created_at REAL
    );
    CREATE TABLE settings_refs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT
    );
    CREATE TABLE state_transitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        ts REAL,
        from_state TEXT,
        to_state TEXT,
        reason TEXT
    );
    """,
    # 2: Pipeline V2 (additive only). Every existing run is a V1 run through the column
    #    default; no existing row or column is rewritten. Code without V2 ignores both columns.
    """
    ALTER TABLE runs ADD COLUMN pipeline_version TEXT NOT NULL DEFAULT 'v1';
    ALTER TABLE runs ADD COLUMN pipeline_state_json TEXT NOT NULL DEFAULT '{}';
    """,
    # 3: V5 worker profiles (additive only). worker_snapshot_json is the run's worker-profile snapshot, written ONCE
    #    before the run's first worker launch; '{}' (every existing row) = a V4 run, bound to the legacy profile at
    #    its first V5 resume. worker_calls records the profile each call ran with (NULL = before V5).
    #    Worker profiles themselves are configuration and live in settings.json, not here.
    """
    ALTER TABLE runs ADD COLUMN worker_snapshot_json TEXT NOT NULL DEFAULT '{}';
    ALTER TABLE worker_calls ADD COLUMN worker_profile_id TEXT;
    ALTER TABLE worker_calls ADD COLUMN worker_profile_revision TEXT;
    """,
]


def migrate(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    current = row[0] if row else 0
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
    for i, script in enumerate(MIGRATIONS[current:], start=current + 1):
        conn.executescript("BEGIN;" + script + f"; UPDATE schema_version SET version = {i}; COMMIT;")
    conn.commit()
    return len(MIGRATIONS)
