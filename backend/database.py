"""
ISL Connect — Database configuration and initialisation.

Usage:
    from backend.database import get_db, init_db

    # In an application startup handler:
    init_db()

    # In a FastAPI dependency:
    db = next(get_db())
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ---------------------------------------------------------------------------
# Resolve database path
# ---------------------------------------------------------------------------
# DATABASE_PATH may be overridden via the .env file.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_DB_PATH = _PROJECT_ROOT / "database" / "isl_connect.db"
_DB_PATH = Path(os.getenv("DATABASE_PATH", str(_DEFAULT_DB_PATH)))

# Ensure the containing directory exists before SQLAlchemy tries to open it.
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{_DB_PATH}"

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    # Required for SQLite to honour foreign-key constraints.
    connect_args={"check_same_thread": False},
    # Echo SQL statements to stdout only in debug mode.
    echo=bool(os.getenv("DEBUG", "false").lower() == "true"),
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record):
    """Enable foreign-key enforcement on every new SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    Base class for all ORM models.

    Import this in each model module:
        from backend.database import Base
    """
    pass


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_db():
    """
    FastAPI dependency that yields a database session and always closes it.

    Example::

        from fastapi import Depends
        from backend.database import get_db

        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all tables defined in ORM models if they do not already exist.

    This function is idempotent — it is safe to call it multiple times
    (e.g., on every application start-up).  It will never drop or overwrite
    existing data.

    Call order:
        1. Import all model modules so their classes are registered on ``Base``.
        2. Call ``Base.metadata.create_all(engine)`` to emit CREATE TABLE IF
           NOT EXISTS statements.
    """
    # Import model modules to register them with Base.metadata before
    # create_all is called.  Add new model imports here as the schema grows.
    import backend.models  # noqa: F401  (side-effect import)

    Base.metadata.create_all(bind=engine)

    # Verify the connection works.
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1, "Database sanity check failed."

    print(f"[database] Initialised successfully -> {_DB_PATH}")
