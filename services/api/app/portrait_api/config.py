from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


CsvList = Annotated[list[str], NoDecode, BeforeValidator(_split_csv)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    expose_api_docs: bool = False
    expose_metrics: bool = False
    rate_limit_fail_closed: bool = True
    app_base_url: str = "http://localhost:3000"
    api_base_url: str = "http://localhost:8000"
    public_api_base_url: str = "/api/v1"
    database_url: str = "postgresql+psycopg://portrait:portrait@postgres:5432/portrait"
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    s3_endpoint_url: str | None = "http://minio:9000"
    s3_region: str = "us-east-1"
    s3_access_key_id: SecretStr = SecretStr("portrait")
    s3_secret_access_key: SecretStr = SecretStr("portrait-local-only")
    s3_bucket: str = "portrait-style-transfer"
    s3_force_path_style: bool = True
    s3_server_side_encryption: Literal["none", "AES256", "aws:kms"] = "none"
    s3_kms_key_id: str | None = None
    signed_url_ttl_seconds: int = Field(default=300, ge=60, le=3600)

    asset_ttl_hours: int = Field(default=24, ge=1, le=720)
    max_upload_bytes: int = Field(default=15 * 1024 * 1024, ge=1024)
    max_decoded_pixels: int = Field(default=8_000_000, ge=1_000_000)
    max_original_long_edge: int = Field(default=8000, ge=512)
    max_session_upload_bytes_24h: int = Field(default=100 * 1024 * 1024, ge=1024)
    max_concurrent_jobs_per_user: int = Field(default=2, ge=1, le=20)
    max_assets_per_session: int = Field(default=100, ge=1, le=10_000)

    model_dir: Path = Path("/models")
    face_landmarker_model: str = "face_landmarker.task"
    image_segmenter_model: str = "selfie_multiclass_256x256.tflite"
    model_manifest_file: str = "manifest.json"
    require_models_for_readiness: bool = True
    allow_heuristic_analyzer: bool = False
    enable_gpu: bool = False
    dense_alignment_device: str = "cpu"
    default_algorithm_profile: Literal["ai_dgpst_v1"] = "ai_dgpst_v1"
    enable_source_compat_profile: bool = False

    cors_origins: CsvList = ["http://localhost:3000"]
    session_secret: SecretStr = SecretStr("development-only-change-me")
    session_cookie_name: str = "pst_session"
    csrf_cookie_name: str = "pst_csrf"
    download_cookie_name: str = "pst_download"
    session_max_age_seconds: int = Field(default=7 * 24 * 3600, ge=3600)
    cookie_domain: str | None = None
    cookie_secure_override: bool | None = None

    rate_limit_requests_per_minute: int = Field(default=120, ge=1)
    rate_limit_uploads_per_hour: int = Field(default=20, ge=1)
    rate_limit_jobs_per_hour: int = Field(default=30, ge=1)
    redis_progress_ttl_seconds: int = Field(default=48 * 3600, ge=3600)
    job_lock_ttl_seconds: int = Field(default=3600, ge=60)
    worker_task_time_limit_seconds: int = Field(default=1800, ge=60)
    max_job_cache_bytes: int = Field(default=512 * 1024 * 1024, ge=1024 * 1024)
    ai_engine_url: str = "http://ai-engine:8010"
    ai_engine_request_timeout_seconds: float = Field(default=600.0, ge=30.0, le=3600.0)
    ai_engine_api_token: SecretStr | None = None

    output_signed_urls_in_diagnostics: bool = True
    log_level: str = "INFO"
    otel_exporter_otlp_endpoint: str | None = None
    algorithm_version: str = "0.1.0"
    auto_create_schema: bool = False
    initialize_storage_on_startup: bool = True

    @property
    def cookie_secure(self) -> bool:
        if self.cookie_secure_override is not None:
            return self.cookie_secure_override
        return self.app_env == "production"

    @property
    def required_model_paths(self) -> tuple[Path, Path]:
        return (
            self.model_dir / self.face_landmarker_model,
            self.model_dir / self.image_segmenter_model,
        )

    @model_validator(mode="after")
    def validate_production_security(self) -> Settings:
        if self.app_env == "production":
            if self.expose_api_docs or self.expose_metrics:
                raise ValueError("API docs and metrics cannot be exposed in production")
            session_secret = self.session_secret.get_secret_value()
            if session_secret == "development-only-change-me" or len(session_secret.strip()) < 32:
                raise ValueError("SESSION_SECRET must contain at least 32 characters in production")
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS is forbidden in production")
            if not self.cookie_secure:
                raise ValueError("Secure cookies are required in production")
            if self.allow_heuristic_analyzer:
                raise ValueError("The heuristic portrait analyzer is forbidden in production")
            if self.s3_server_side_encryption == "none":
                raise ValueError("Server-side object encryption is required in production")
            if self.s3_server_side_encryption == "aws:kms" and not self.s3_kms_key_id:
                raise ValueError("S3_KMS_KEY_ID is required for aws:kms encryption")
            ai_token = (
                self.ai_engine_api_token.get_secret_value().strip()
                if self.ai_engine_api_token is not None
                else ""
            )
            if len(ai_token) < 32:
                raise ValueError(
                    "AI_ENGINE_API_TOKEN must contain at least 32 characters in production"
                )
        if self.enable_gpu and not self.dense_alignment_device.lower().startswith("cuda"):
            raise ValueError("ENABLE_GPU requires DENSE_ALIGNMENT_DEVICE=cuda")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
