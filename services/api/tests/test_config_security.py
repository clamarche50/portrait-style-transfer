from __future__ import annotations

import pytest
from portrait_api.config import Settings
from pydantic import ValidationError


def test_production_session_secret_requires_at_least_32_characters() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(
            app_env="production",
            session_secret="changed-but-too-short",
            s3_server_side_encryption="AES256",
            ai_engine_api_token="an-internal-ai-token-with-32-characters",
        )

    with pytest.raises(ValidationError, match="AI_ENGINE_API_TOKEN"):
        Settings(
            app_env="production",
            session_secret="a-production-secret-with-32-chars-minimum",
            s3_server_side_encryption="AES256",
        )

    settings = Settings(
        app_env="production",
        session_secret="a-production-secret-with-32-chars-minimum",
        s3_server_side_encryption="AES256",
        ai_engine_api_token="an-internal-ai-token-with-32-characters",
    )
    assert settings.cookie_secure


def test_environment_accepts_documented_comma_separated_cors_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "https://studio.example, http://localhost:3000",
    )

    settings = Settings(_env_file=None)

    assert settings.cors_origins == [
        "https://studio.example",
        "http://localhost:3000",
    ]
