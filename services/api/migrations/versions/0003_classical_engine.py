"""Remap InstantStyle engine rows onto the classical 2014 engine.

Revision ID: 0003_classical_engine
Revises: 0002_instantstyle_engine
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_classical_engine"
down_revision: str | None = "0002_instantstyle_engine"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INSTANTSTYLE_PROFILES = ("ai_instantstyle_v1", "ai_dgpst_v1")


def upgrade() -> None:
    jobs = sa.table(
        "jobs",
        sa.column("algorithm_profile", sa.String),
        sa.column("stage", sa.String),
        sa.column("status", sa.String),
        sa.column("settings", sa.JSON),
    )
    op.execute(
        jobs.update()
        .where(jobs.c.algorithm_profile.in_(_INSTANTSTYLE_PROFILES))
        .values(algorithm_profile="source_2014_compat")
    )
    # The AI_GENERATION stage does not exist in the classical engine; fold
    # finished rows onto COMPLETED and restart anything still in flight.
    op.execute(
        jobs.update()
        .where(sa.and_(jobs.c.stage == "AI_GENERATION", jobs.c.status == "SUCCEEDED"))
        .values(stage="COMPLETED")
    )
    op.execute(
        jobs.update()
        .where(sa.and_(jobs.c.stage == "AI_GENERATION", jobs.c.status != "SUCCEEDED"))
        .values(stage="VALIDATING")
    )
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            UPDATE jobs
            SET settings = jsonb_set(
                settings, '{algorithm_profile}', '"source_2014_compat"'
            )
            WHERE settings->>'algorithm_profile' IN ('ai_instantstyle_v1', 'ai_dgpst_v1')
            """
        )


def downgrade() -> None:
    # InstantStyle engine rows cannot be reconstructed; nothing to restore.
    pass
