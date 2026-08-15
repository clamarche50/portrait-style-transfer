"""Strict HTTP client for the local portrait AI engine.

The argument names in this module are deliberately domain-specific.  A content
portrait owns identity and geometry; a style portrait supplies appearance.  Do
not collapse them into positional ``input``/``reference`` arguments at this
boundary because swapping the images produces a plausible but incorrect result.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import httpx

AI_ENGINE_ID = "ai_instantstyle_v1"
_DEFAULT_BASE_URL = "http://ai-engine:8010"
_DEFAULT_TIMEOUT_SECONDS = 600.0
_MAX_OUTPUT_BYTES = 96 * 1024 * 1024
_MAX_DIAGNOSTICS_BYTES = 32 * 1024
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class AIEngineResponse:
    """A validated response from the AI engine."""

    image_png: bytes
    diagnostics: dict[str, Any]
    engine_id: str


class AIEngineError(RuntimeError):
    """Safe, classified failure returned by or while calling the AI engine."""

    def __init__(self, code: str, safe_message: str, *, retryable: bool) -> None:
        super().__init__(safe_message)
        self.code = code if _ERROR_CODE.fullmatch(code) else "AI_ENGINE_ERROR"
        self.safe_message = safe_message[:500]
        self.retryable = retryable


def _decode_diagnostics(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    if len(value) > _MAX_DIAGNOSTICS_BYTES * 2:
        raise AIEngineError(
            "AI_ENGINE_INVALID_RESPONSE",
            "The AI engine returned invalid diagnostics.",
            retryable=False,
        )
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = base64.b64decode(padded, altchars=b"-_", validate=True)
        if len(payload) > _MAX_DIAGNOSTICS_BYTES:
            raise ValueError("diagnostics are too large")
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("diagnostics must be an object")
        return {str(key): item for key, item in decoded.items()}
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise AIEngineError(
            "AI_ENGINE_INVALID_RESPONSE",
            "The AI engine returned invalid diagnostics.",
            retryable=False,
        ) from exc


def _error_detail(response: httpx.Response) -> tuple[str, str, bool]:
    retryable_default = response.status_code in {408, 425, 429} or response.status_code >= 500
    code = "AI_ENGINE_UNAVAILABLE" if retryable_default else "AI_ENGINE_REQUEST_REJECTED"
    message = (
        "The AI engine is temporarily unavailable."
        if retryable_default
        else "The AI engine rejected the transfer request."
    )
    retryable = retryable_default
    try:
        detail: Any = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return code, message, retryable

    if isinstance(detail, dict) and "detail" in detail:
        detail = detail["detail"]
    if isinstance(detail, dict) and "detail" in detail:
        detail = detail["detail"]
    if not isinstance(detail, dict):
        return code, message, retryable

    remote_code = detail.get("code")
    remote_message = detail.get("message")
    remote_retryable = detail.get("retryable")
    if isinstance(remote_code, str) and _ERROR_CODE.fullmatch(remote_code):
        code = remote_code
    if isinstance(remote_message, str) and remote_message.strip():
        message = remote_message.strip()[:500]
    if isinstance(remote_retryable, bool):
        retryable = remote_retryable
    return code, message, retryable


def _image_part(role: str, payload: bytes) -> tuple[str, bytes, str]:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        extension, media_type = "png", "image/png"
    elif payload.startswith(b"\xff\xd8\xff"):
        extension, media_type = "jpg", "image/jpeg"
    elif payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        extension, media_type = "webp", "image/webp"
    else:
        extension, media_type = "bin", "application/octet-stream"
    return f"{role}.{extension}", payload, media_type


class AIEngineClient:
    """Synchronous client used by a Celery worker process."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        api_token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0)),
            transport=transport,
            headers={"Accept": "image/png", "User-Agent": "portrait-style-worker/0.1"},
        )
        self._api_token = api_token or None

    def __repr__(self) -> str:
        return (
            f"AIEngineClient(base_url={self._client.base_url!s}, "
            f"token_configured={bool(self._api_token)})"
        )

    def matches_api_token(self, api_token: str | None) -> bool:
        expected = self._api_token or ""
        candidate = api_token or ""
        return hmac.compare_digest(expected, candidate)

    def close(self) -> None:
        self._client.close()

    def transfer(
        self,
        *,
        content: bytes,
        style: bytes,
        settings: Mapping[str, object],
    ) -> AIEngineResponse:
        """Transfer ``style`` appearance while preserving ``content`` identity."""

        if not content:
            raise ValueError("content image is empty")
        if not style:
            raise ValueError("style image is empty")
        try:
            settings_json = json.dumps(
                dict(settings),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("settings must be JSON serializable") from exc

        try:
            headers = (
                {"Authorization": f"Bearer {self._api_token}"}
                if self._api_token is not None
                else None
            )
            response = self._client.post(
                "/v1/transfer",
                data={"settings": settings_json},
                files={
                    "content": _image_part("content", content),
                    "style": _image_part("style", style),
                },
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise AIEngineError(
                "AI_ENGINE_TIMEOUT",
                "The AI engine timed out while processing the portrait.",
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise AIEngineError(
                "AI_ENGINE_UNAVAILABLE",
                "The AI engine is temporarily unavailable.",
                retryable=True,
            ) from exc

        if not response.is_success:
            code, message, retryable = _error_detail(response)
            raise AIEngineError(code, message, retryable=retryable)

        engine_id = response.headers.get("X-Portrait-Engine", "")
        if engine_id != AI_ENGINE_ID:
            raise AIEngineError(
                "AI_ENGINE_INVALID_RESPONSE",
                "The AI engine returned an unexpected engine identifier.",
                retryable=False,
            )
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "image/png" or not response.content:
            raise AIEngineError(
                "AI_ENGINE_INVALID_RESPONSE",
                "The AI engine did not return a PNG image.",
                retryable=False,
            )
        if len(response.content) > _MAX_OUTPUT_BYTES:
            raise AIEngineError(
                "AI_ENGINE_INVALID_RESPONSE",
                "The AI engine output exceeded the worker limit.",
                retryable=False,
            )

        diagnostics = _decode_diagnostics(response.headers.get("X-Portrait-Diagnostics"))
        return AIEngineResponse(
            image_png=bytes(response.content),
            diagnostics=diagnostics,
            engine_id=engine_id,
        )


def _configured_value(settings: object, name: str, env_name: str, default: object) -> object:
    value = getattr(settings, name, None)
    if value is not None:
        return value
    return os.getenv(env_name, default)


_CLIENTS: dict[tuple[str, float], AIEngineClient] = {}
_CLIENTS_LOCK = threading.Lock()


def get_ai_engine_client(settings: object) -> AIEngineClient:
    """Build a process-local pooled client from shared application settings."""

    base_url = str(_configured_value(settings, "ai_engine_url", "AI_ENGINE_URL", _DEFAULT_BASE_URL))
    timeout = float(
        cast(
            Any,
            _configured_value(
                settings,
                "ai_engine_request_timeout_seconds",
                "AI_ENGINE_REQUEST_TIMEOUT_SECONDS",
                _configured_value(
                    settings,
                    "ai_engine_timeout_seconds",
                    "AI_ENGINE_TIMEOUT_SECONDS",
                    _DEFAULT_TIMEOUT_SECONDS,
                ),
            ),
        )
    )
    configured_token = _configured_value(
        settings,
        "ai_engine_api_token",
        "AI_ENGINE_API_TOKEN",
        "",
    )
    get_secret_value = getattr(configured_token, "get_secret_value", None)
    token = str(get_secret_value() if callable(get_secret_value) else configured_token) or None
    cache_key = (base_url, timeout)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(cache_key)
        if client is None or not client.matches_api_token(token):
            if client is not None:
                client.close()
            client = AIEngineClient(
                base_url=base_url,
                timeout_seconds=timeout,
                api_token=token,
            )
            _CLIENTS[cache_key] = client
        return client
