"""Create the API domain tables.

Revision ID: 0001_initial
Revises: None
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_table(
        "assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=True),
        sa.Column("session_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_assets_owner_id_users", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_assets"),
        sa.UniqueConstraint("object_key", name="uq_assets_object_key"),
    )
    op.create_index("ix_assets_expiry_active", "assets", ["expires_at", "deleted_at"])
    op.create_index("ix_assets_kind", "assets", ["kind"])
    op.create_index("ix_assets_session_active", "assets", ["session_id", "deleted_at"])
    op.create_index("ix_assets_session_id", "assets", ["session_id"])
    op.create_index("ix_assets_sha256", "assets", ["sha256"])

    op.create_table(
        "styles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=True),
        sa.Column("session_id", sa.Uuid(), nullable=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=2000), nullable=False),
        sa.Column("rights_confirmed", sa.Boolean(), nullable=False),
        sa.Column("is_public", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_styles_owner_id_users", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_styles"),
    )
    op.create_index("ix_styles_session_active", "styles", ["session_id", "deleted_at"])
    op.create_index("ix_styles_session_id", "styles", ["session_id"])

    op.create_table(
        "style_examples",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("style_id", sa.Uuid(), nullable=False),
        sa.Column("asset_id", sa.Uuid(), nullable=False),
        sa.Column("feature_object_key", sa.String(length=1024), nullable=True),
        sa.Column("quality", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
            name="fk_style_examples_asset_id_assets",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["style_id"],
            ["styles.id"],
            name="fk_style_examples_style_id_styles",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_style_examples"),
        sa.UniqueConstraint("style_id", "asset_id", name="uq_style_examples_style_asset"),
    )
    op.create_index("ix_style_examples_style_id", "style_examples", ["style_id"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=True),
        sa.Column("session_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=48), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("input_asset_id", sa.Uuid(), nullable=False),
        sa.Column("reference_asset_id", sa.Uuid(), nullable=True),
        sa.Column("style_id", sa.Uuid(), nullable=True),
        sa.Column("selected_style_example_id", sa.Uuid(), nullable=True),
        sa.Column("algorithm_profile", sa.String(length=32), nullable=False),
        sa.Column("settings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("corrections", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("diagnostics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message_safe", sa.String(length=500), nullable=True),
        sa.Column("worker_id", sa.String(length=255), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(reference_asset_id IS NULL) <> (style_id IS NULL)",
            name="ck_jobs_exactly_one_reference_or_style",
        ),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_jobs_valid_progress"),
        sa.ForeignKeyConstraint(
            ["input_asset_id"],
            ["assets.id"],
            name="fk_jobs_input_asset_id_assets",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_jobs_owner_id_users", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["reference_asset_id"],
            ["assets.id"],
            name="fk_jobs_reference_asset_id_assets",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["selected_style_example_id"],
            ["style_examples.id"],
            name="fk_jobs_selected_style_example_id_style_examples",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["style_id"], ["styles.id"], name="fk_jobs_style_id_styles", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
    )
    op.create_index("ix_jobs_expiry_status", "jobs", ["expires_at", "status"])
    op.create_index("ix_jobs_session_id", "jobs", ["session_id"])
    op.create_index("ix_jobs_session_status", "jobs", ["session_id", "status"])
    op.create_index("ix_jobs_status", "jobs", ["status"])

    op.create_table(
        "job_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("asset_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_kind", sa.String(length=48), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["asset_id"], ["assets.id"], name="fk_job_artifacts_asset_id_assets", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_job_artifacts_job_id_jobs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_job_artifacts"),
        sa.UniqueConstraint("job_id", "asset_id", name="uq_job_artifacts_job_asset"),
    )
    op.create_index("ix_job_artifacts_job_id", "job_artifacts", ["job_id"])


def downgrade() -> None:
    op.drop_table("job_artifacts")
    op.drop_table("jobs")
    op.drop_table("style_examples")
    op.drop_table("styles")
    op.drop_table("assets")
    op.drop_table("users")
