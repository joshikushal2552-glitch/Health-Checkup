"""
SQLAlchemy engine/session factory for the imaging-relational layer
(api/models.py).

WHY A SEPARATE PHYSICAL DATABASE FROM db.py, FOR NOW
    db.py's stdlib sqlite3 connections and SQLAlchemy's own connection pool
    are two different drivers. Pointed at the same SQLite file, concurrent
    writes from both can hit "database is locked" under SQLite's default
    journal mode. Rather than tune journal modes for a transitional setup,
    local SQLite development uses a second file (instance/imaging.db,
    alongside db.py's vitalitysync.db) by default.

    This split is dev-only and deliberately temporary. IMAGING_DATABASE_URL
    defaults to sqlite:///<instance>/imaging.db for local development; when
    it is pointed at a real Postgres instance (the ARCHITECTURE_AUDIT.md
    section 4 target), db.py's tables can move into the same Postgres
    database in a later change without touching this module's public
    surface (get_session / init_models_db).

ENVIRONMENT
    IMAGING_DATABASE_URL - a full SQLAlchemy URL, e.g.
        postgresql+psycopg2://user:pass@host/dbname
        sqlite:////absolute/path/to/imaging.db
    When unset, defaults to a SQLite file under the same instance/
    directory study_store.py and db.py already use.
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from models import Base

_INSTANCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instance"
)

DEFAULT_IMAGING_DATABASE_URL = "sqlite:///" + os.path.join(_INSTANCE_DIR, "imaging.db")


def _resolve_url(url=None):
    return url or os.environ.get("IMAGING_DATABASE_URL", DEFAULT_IMAGING_DATABASE_URL)


_engine = None
_engine_url = None
_SessionLocal = None


def get_engine(url=None):
    """Returns the module-level engine, creating it (or recreating it if the
    resolved URL changed - relevant for tests that set the env var per
    case) as needed."""
    global _engine, _engine_url, _SessionLocal
    resolved = _resolve_url(url)
    if _engine is not None and _engine_url == resolved:
        return _engine
    if resolved.startswith("sqlite"):
        # Create the parent directory of whatever file the resolved URL
        # actually points to - not the hardcoded default _INSTANCE_DIR, which
        # is read-only on a serverless deployment once IMAGING_DATABASE_URL
        # has been overridden to point elsewhere (e.g. /tmp on Vercel).
        db_file = make_url(resolved).database
        if db_file and db_file != ":memory:":
            parent_dir = os.path.dirname(db_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        _engine = create_engine(resolved, connect_args={"check_same_thread": False})
    else:
        _engine = create_engine(resolved, pool_pre_ping=True)
    _engine_url = resolved
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_models_db(url=None):
    """Creates all tables that don't already exist. Safe to call repeatedly
    (mirrors db.init_db's CREATE TABLE IF NOT EXISTS behaviour). Real schema
    evolution beyond the first deployment should go through Alembic
    (migrations/), not this function - see DATABASE_DESIGN.md."""
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def get_session(url=None):
    """Context-managed session: commits on clean exit, rolls back on
    exception, always closes."""
    get_engine(url)  # ensures _SessionLocal is bound to the resolved engine
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine():
    """Closes the current engine's connection pool. Test-support only, so a
    suite that rotates through many throwaway SQLite files doesn't leak
    open file descriptors between cases."""
    global _engine, _engine_url, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _engine_url = None
    _SessionLocal = None


def reset_models_db(url=None):
    """Drops and recreates every table. Test-support only, mirrors
    db.reset_db."""
    engine = get_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine
