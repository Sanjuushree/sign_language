"""
ISL Connect — ORM Models (Phase 1 skeleton).

Only the minimum schema is defined here.  Additional tables (e.g. Signs,
Sessions, Predictions) will be added in later phases once the ML pipeline
is ready.

All models inherit from ``backend.database.Base`` so that
``Base.metadata.create_all()`` picks them up automatically.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


# ---------------------------------------------------------------------------
# SystemEvent — lightweight audit / event log
# ---------------------------------------------------------------------------

class SystemEvent(Base):
    """
    Records significant system events (start-up, errors, etc.).

    This table is intentionally minimal for Phase 1 and is primarily here
    to verify that the database layer is working end-to-end.
    """

    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    is_error: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<SystemEvent id={self.id} type={self.event_type!r} "
            f"created_at={self.created_at}>"
        )


# ---------------------------------------------------------------------------
# Future models (placeholders — implement in later phases)
# ---------------------------------------------------------------------------
# class Sign(Base):
#     """Represents a known ISL sign in the vocabulary."""
#     __tablename__ = "signs"
#     ...
#
# class RecognitionSession(Base):
#     """Records a live recognition session and its outcomes."""
#     __tablename__ = "recognition_sessions"
#     ...
