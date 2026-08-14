from __future__ import annotations

import hashlib
import hmac
import uuid
from urllib.parse import urlsplit

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from portrait_api.config import Settings
from starlette.responses import Response


class AnonymousSessionSigner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        secret = settings.session_secret.get_secret_value()
        self._signer = TimestampSigner(secret, salt="portrait-anonymous-session")
        self._csrf_key = secret.encode()

    def issue(self, session_id: uuid.UUID) -> str:
        return self._signer.sign(str(session_id)).decode()

    def verify(self, value: str | None) -> uuid.UUID | None:
        if not value:
            return None
        try:
            unsigned = self._signer.unsign(
                value,
                max_age=self.settings.session_max_age_seconds,
            ).decode()
            return uuid.UUID(unsigned)
        except (BadSignature, SignatureExpired, UnicodeDecodeError, ValueError):
            return None

    def csrf_token(self, session_id: uuid.UUID) -> str:
        return hmac.new(
            self._csrf_key,
            f"csrf:{session_id}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def valid_csrf(self, session_id: uuid.UUID, cookie: str | None, header: str | None) -> bool:
        if not cookie or not header:
            return False
        expected = self.csrf_token(session_id)
        return hmac.compare_digest(cookie, expected) and hmac.compare_digest(header, expected)


class DownloadTokenSigner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._signer = TimestampSigner(
            settings.session_secret.get_secret_value(),
            salt="portrait-asset-download",
        )

    def issue(self, asset_id: uuid.UUID, session_id: uuid.UUID) -> str:
        return self._signer.sign(f"{asset_id}:{session_id}").decode()

    def cookie_path(self, asset_id: uuid.UUID) -> str:
        base_path = urlsplit(self.settings.public_api_base_url).path.rstrip("/")
        return f"{base_path}/assets/{asset_id}/content"

    def set_cookie(
        self,
        response: Response,
        asset_id: uuid.UUID,
        session_id: uuid.UUID,
    ) -> None:
        response.set_cookie(
            self.settings.download_cookie_name,
            self.issue(asset_id, session_id),
            max_age=self.settings.signed_url_ttl_seconds,
            httponly=True,
            secure=self.settings.cookie_secure,
            samesite="strict",
            domain=self.settings.cookie_domain,
            path=self.cookie_path(asset_id),
        )

    def valid(
        self,
        token: str | None,
        asset_id: uuid.UUID,
        session_id: uuid.UUID,
    ) -> bool:
        if not token:
            return False
        try:
            unsigned = self._signer.unsign(
                token,
                max_age=self.settings.signed_url_ttl_seconds,
            ).decode()
        except (BadSignature, SignatureExpired, UnicodeDecodeError):
            return False
        return hmac.compare_digest(unsigned, f"{asset_id}:{session_id}")


def session_hash(session_id: uuid.UUID) -> str:
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:16]
