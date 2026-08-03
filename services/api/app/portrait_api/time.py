from __future__ import annotations

from datetime import UTC, datetime


def as_utc(value: datetime) -> datetime:
    """Normalize timestamps returned by SQLite and PostgreSQL to aware UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def is_expired(value: datetime, *, now: datetime | None = None) -> bool:
    return as_utc(value) <= (now or datetime.now(UTC))
