"""Remap legacy engine enum values onto the InstantStyle engine.

Revision ID: 0002_instantstyle_engine
Revises: 0001_initial
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_instantstyle_engine"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEGACY_STAGES = (
    "AFFINE_ALIGNMENT",
    "PIECEWISE_ALIGNMENT",
    "DENSE_ALIGNMENT",
    "MULTISCALE_TRANSFER",
    "EYE_HIGHLIGHTS",
)
_LEGACY_ARTIFACT_KINDS = (
    "AFFINE_PREVIEW",
    "PIECEWISE_PREVIEW",
    "DENSE_PREVIEW",
    "ENERGY",
    "GAIN",
)


def upgrade() -> None:
    jobs = sa.table(
        "jobs",
        sa.column("algorithm_profile", sa.String),
        sa.column("stage", sa.String),
    )
    op.execute(
        jobs.update()
        .where(jobs.c.algorithm_profile.in_(("ai_dgpst_v1", "paper_exact", "source_2014_compat")))
        .values(algorithm_profile="ai_instantstyle_v1")
    )
    op.execute(jobs.update().where(jobs.c.stage.in_(_LEGACY_STAGES)).values(stage="AI_GENERATION"))
    artifacts = sa.table(
        "job_artifacts",
        sa.column("artifact_kind", sa.String),
    )
    op.execute(
        artifacts.update()
        .where(artifacts.c.artifact_kind.in_(_LEGACY_ARTIFACT_KINDS))
        .values(artifact_kind="OTHER")
    )


def downgrade() -> None:
    # Legacy engine rows cannot be reconstructed; nothing to restore.
    pass
