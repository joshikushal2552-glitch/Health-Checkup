"""
SQLite persistence for doctor accounts, cases, notes, and the audit log.

Uses the Python standard library `sqlite3` only - no ORM - to keep the
deployed serverless function small and avoid adding heavy dependencies to a
project that previously had no database at all.

DEPLOYMENT LIMITATION (documented, not worked around):
    On Vercel's serverless Python runtime the filesystem is ephemeral and not
    shared between function instances, so a SQLite file under /tmp lives only
    as long as one instance. Accounts/cases created on one invocation may not
    be visible on another. This mirrors the pre-existing limitation of the
    in-memory STUDIES dict in api/index.py. For a real multi-user deployment
    this module's connection target should be pointed at a managed database;
    the schema and queries here are deliberately plain SQL to make that
    swap straightforward.

Nothing in this module stores DICOM pixel data or patient-identifying
imaging metadata. Cases reference imaging by opaque study id only.
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone

# Default to a local file next to the repo for development; override with
# DATABASE_PATH (e.g. /tmp/vitalitysync.db on a serverless host).
DEFAULT_DB_PATH = os.environ.get(
    "DATABASE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vitalitysync.db"),
)

_local = threading.local()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_db(path=None):
    """Per-thread connection. Flask's dev server and test client are threaded,
    and sqlite3 connections are not safe to share across threads."""
    db_path = path or DEFAULT_DB_PATH
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == db_path:
        return conn
    if conn is not None:
        conn.close()
    parent_dir = os.path.dirname(db_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
    except sqlite3.OperationalError as exc:
        raise sqlite3.OperationalError(
            f"{exc} (resolved DATABASE_PATH={db_path!r}; "
            f"DATABASE_PATH env var={os.environ.get('DATABASE_PATH')!r})"
        ) from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _local.conn = conn
    _local.path = db_path
    return conn


def close_db():
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
        _local.path = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    email          TEXT NOT NULL UNIQUE,
    password_hash  TEXT NOT NULL,
    display_name   TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cases (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    case_ref          TEXT NOT NULL UNIQUE,
    owner_doctor_id   INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    title             TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'needs_review',
    is_demo           INTEGER NOT NULL DEFAULT 0,
    study_id          TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- Explicit per-doctor grants. Ownership alone is not consulted by the
-- authorization check; the owner is also inserted here at creation time so
-- there is exactly one code path for "may this doctor see this case".
CREATE TABLE IF NOT EXISTS case_access (
    case_id    INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    doctor_id  INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    role       TEXT NOT NULL DEFAULT 'viewer',
    granted_at TEXT NOT NULL,
    PRIMARY KEY (case_id, doctor_id)
);

CREATE TABLE IF NOT EXISTS notes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id           INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    author_doctor_id  INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    content           TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- Security-relevant events only. Deliberately does NOT store note bodies,
-- imaging content, or any clinical detail - just who did what to which
-- opaque identifier, and when.
CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    doctor_id    INTEGER,
    event        TEXT NOT NULL,
    target_type  TEXT,
    target_id    TEXT,
    outcome      TEXT NOT NULL DEFAULT 'success',
    ip           TEXT
);

CREATE INDEX IF NOT EXISTS idx_cases_owner ON cases(owner_doctor_id);
CREATE INDEX IF NOT EXISTS idx_case_access_doctor ON case_access(doctor_id);
CREATE INDEX IF NOT EXISTS idx_notes_case ON notes(case_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""


def init_db(path=None):
    conn = get_db(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def reset_db(path=None):
    """Drops and recreates every table. Test-support only."""
    conn = get_db(path)
    for table in ("audit_log", "notes", "case_access", "cases", "doctors"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def record_audit(event, doctor_id=None, target_type=None, target_id=None,
                  outcome="success", ip=None, path=None):
    """Appends a security-relevant event.

    Callers must pass only non-sensitive identifiers. Never pass note text,
    patient metadata, or imaging content.
    """
    conn = get_db(path)
    conn.execute(
        "INSERT INTO audit_log (ts, doctor_id, event, target_type, target_id, outcome, ip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (utc_now_iso(), doctor_id, event, target_type,
         str(target_id) if target_id is not None else None, outcome, ip),
    )
    conn.commit()
