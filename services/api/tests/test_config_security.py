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
        )

    settings = Settings(
        app_env="production",
        session_secret="a-production-secret-with-32-chars-minimum",
        s3_server_side_encryption="AES256",
    )
    assert settings.cookie_secure
