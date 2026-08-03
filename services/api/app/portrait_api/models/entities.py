from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from portrait_api.db.base import Base
from portrait_api.models.enums import (
    AlgorithmProfile,
    ArtifactKind,
    AssetKind,
    JobStatus,
    ProcessingStage,
)
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

JsonType = JSON().with_variant(JSONB(), "postgresql")


def enum_type(enum: type[StrEnum], name: str) -> Enum:
    return Enum(
        enum,
        name=name,
        native_enum=False,
        values_callable=lambda cls: [item.value for item in cls],
        validate_strings=True,
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str | None] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Asset(Base):
    __tablename__ = "assets"
    __table_args__ = (
        Index("ix_assets_session_active", "session_id", "deleted_at"),
        Index("ix_assets_expiry_active", "expires_at", "deleted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    session_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), index=True)
    kind: Mapped[AssetKind] = mapped_column(enum_type(AssetKind, "asset_kind"), index=True)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JsonType, default=dict)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Style(Base):
    __tablename__ = "styles"
    __table_args__ = (Index("ix_styles_session_active", "session_id", "deleted_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    session_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(String(2000), default="")
    rights_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    examples: Mapped[list[StyleExample]] = relationship(
        back_populates="style", cascade="all, delete-orphan", lazy="selectin"
    )


class StyleExample(Base):
    __tablename__ = "style_examples"
    __table_args__ = (
        UniqueConstraint("style_id", "asset_id", name="uq_style_examples_style_asset"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    style_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("styles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False
    )
    feature_object_key: Mapped[str | None] = mapped_column(String(1024))
    quality: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    style: Mapped[Style] = relationship(back_populates="examples")
    asset: Mapped[Asset] = relationship(lazy="joined")


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "(reference_asset_id IS NULL) <> (style_id IS NULL)",
            name="exactly_one_reference_or_style",
        ),
        CheckConstraint("progress >= 0 AND progress <= 100", name="valid_progress"),
        Index("ix_jobs_session_status", "session_id", "status"),
        Index("ix_jobs_expiry_status", "expires_at", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    session_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), index=True)
    status: Mapped[JobStatus] = mapped_column(enum_type(JobStatus, "job_status"), index=True)
    stage: Mapped[ProcessingStage] = mapped_column(enum_type(ProcessingStage, "processing_stage"))
    progress: Mapped[int] = mapped_column(Integer, default=0)
    input_asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id", ondelete="RESTRICT"))
    reference_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("assets.id", ondelete="RESTRICT")
    )
    style_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("styles.id", ondelete="RESTRICT"))
    selected_style_example_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("style_examples.id", ondelete="SET NULL")
    )
    algorithm_profile: Mapped[AlgorithmProfile] = mapped_column(
        enum_type(AlgorithmProfile, "algorithm_profile"), default=AlgorithmProfile.PAPER_EXACT
    )
    settings: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    corrections: Mapped[list[dict[str, Any]] | None] = mapped_column(JsonType)
    diagnostics: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message_safe: Mapped[str | None] = mapped_column(String(500))
    worker_id: Mapped[str | None] = mapped_column(String(255))
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    input_asset: Mapped[Asset] = relationship(foreign_keys=[input_asset_id], lazy="joined")
    reference_asset: Mapped[Asset | None] = relationship(
        foreign_keys=[reference_asset_id], lazy="joined"
    )
    style: Mapped[Style | None] = relationship(lazy="joined")
    selected_style_example: Mapped[StyleExample | None] = relationship(lazy="joined")
    artifacts: Mapped[list[JobArtifact]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )


class JobArtifact(Base):
    __tablename__ = "job_artifacts"
    __table_args__ = (UniqueConstraint("job_id", "asset_id", name="uq_job_artifacts_job_asset"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    artifact_kind: Mapped[ArtifactKind] = mapped_column(
        enum_type(ArtifactKind, "artifact_kind"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    job: Mapped[Job] = relationship(back_populates="artifacts")
    asset: Mapped[Asset] = relationship(lazy="joined")
