from __future__ import annotations

import hashlib
import hmac
import uuid

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from portrait_api.config import Settings


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


def session_hash(session_id: uuid.UUID) -> str:
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:16]
