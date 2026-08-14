from __future__ import annotations

import ipaddress
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import ClassVar

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from portrait_api.config import Settings
from portrait_api.metrics import HTTP_DURATION, HTTP_REQUESTS
from portrait_api.security import AnonymousSessionSigner, session_hash
from portrait_api.services.redis_gateway import ProgressStore
from portrait_api.telemetry import TraceKind, Tracer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

_SAFE_HTTP_METHODS = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"})
_CLOUDFLARE_RAY = re.compile(r"^[0-9a-fA-F]{16,32}-[A-Za-z0-9]{3}$")


def _safe_http_method(method: str) -> str:
    normalized = method.upper()
    return normalized if normalized in _SAFE_HTTP_METHODS else "OTHER"


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return template if isinstance(template, str) and template.startswith("/") else "unmatched"


def _rate_limit_client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    try:
        peer_address = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    cloudflare_ip = request.headers.get("CF-Connecting-IP")
    cloudflare_ray = request.headers.get("CF-Ray", "")
    if (
        (peer_address.is_private or peer_address.is_loopback)
        and cloudflare_ip
        and _CLOUDFLARE_RAY.fullmatch(cloudflare_ray)
    ):
        try:
            return ipaddress.ip_address(cloudflare_ip).compressed
        except ValueError:
            pass
    return peer_address.compressed


def _error(
    request_id: str, status: int, code: str, message: str, **details: object
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
                "request_id": request_id,
            }
        },
    )


class RequestSessionMiddleware(BaseHTTPMiddleware):
    _unsafe_methods: ClassVar[frozenset[str]] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings
        self.signer = AnonymousSessionSigner(settings)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied_request_id = request.headers.get("X-Request-ID")
        try:
            request_id = (
                str(uuid.UUID(supplied_request_id)) if supplied_request_id else str(uuid.uuid4())
            )
        except ValueError:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        signed_cookie = request.cookies.get(self.settings.session_cookie_name)
        session_id = self.signer.verify(signed_cookie)
        is_new = session_id is None
        if session_id is None:
            session_id = uuid.uuid4()
        request.state.session_id = session_id
        request.state.session_hash = session_hash(session_id)

        if (
            request.method in self._unsafe_methods
            and signed_cookie is not None
            and not self.signer.valid_csrf(
                session_id,
                request.cookies.get(self.settings.csrf_cookie_name),
                request.headers.get("X-CSRF-Token"),
            )
        ):
            response: Response = _error(
                request_id,
                403,
                "CSRF_FAILED",
                "The CSRF token is missing or invalid.",
            )
            csrf_token = self.signer.csrf_token(session_id)
            if is_new:
                response.set_cookie(
                    self.settings.session_cookie_name,
                    self.signer.issue(session_id),
                    max_age=self.settings.session_max_age_seconds,
                    httponly=True,
                    secure=self.settings.cookie_secure,
                    samesite="lax",
                    domain=self.settings.cookie_domain,
                    path="/",
                )
            response.set_cookie(
                self.settings.csrf_cookie_name,
                csrf_token,
                max_age=self.settings.session_max_age_seconds,
                httponly=False,
                secure=self.settings.cookie_secure,
                samesite="lax",
                domain=self.settings.cookie_domain,
                path="/",
            )
            response.headers["X-Request-ID"] = request_id
            response.headers["X-CSRF-Token"] = csrf_token
            return response

        response = await call_next(request)
        csrf_token = self.signer.csrf_token(session_id)
        if is_new:
            response.set_cookie(
                self.settings.session_cookie_name,
                self.signer.issue(session_id),
                max_age=self.settings.session_max_age_seconds,
                httponly=True,
                secure=self.settings.cookie_secure,
                samesite="lax",
                domain=self.settings.cookie_domain,
                path="/",
            )
        if request.cookies.get(self.settings.csrf_cookie_name) != csrf_token:
            response.set_cookie(
                self.settings.csrf_cookie_name,
                csrf_token,
                max_age=self.settings.session_max_age_seconds,
                httponly=False,
                secure=self.settings.cookie_secure,
                samesite="lax",
                domain=self.settings.cookie_domain,
                path="/",
            )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-CSRF-Token"] = csrf_token
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, settings: Settings, store: ProgressStore) -> None:
        super().__init__(app)
        self.settings = settings
        self.store = store

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.method == "OPTIONS" or request.url.path in {"/metrics", "/api/v1/health/live"}:
            return await call_next(request)
        path = request.url.path
        if request.method == "POST" and path == "/api/v1/assets/upload":
            limit, window, category = self.settings.rate_limit_uploads_per_hour, 3600, "upload"
        elif request.method == "POST" and path == "/api/v1/jobs":
            limit, window, category = self.settings.rate_limit_jobs_per_hour, 3600, "job"
        else:
            limit, window, category = self.settings.rate_limit_requests_per_minute, 60, "request"
        client_ip = _rate_limit_client_ip(request)
        try:
            session_allowed, session_retry_after = await self.store.rate_limit(
                f"{category}:session:{request.state.session_hash}", limit, window
            )
            ip_allowed, ip_retry_after = await self.store.rate_limit(
                f"{category}:ip:{client_ip}", limit, window
            )
            allowed = session_allowed and ip_allowed
            retry_after = max(session_retry_after, ip_retry_after)
        except Exception:
            if self.settings.rate_limit_fail_closed:
                return _error(
                    request.state.request_id,
                    503,
                    "RATE_LIMIT_UNAVAILABLE",
                    "Request controls are temporarily unavailable.",
                )
            allowed, retry_after = True, 0
        if not allowed:
            response = _error(
                request.state.request_id,
                429,
                "RATE_LIMITED",
                "Too many requests. Try again later.",
                retry_after_seconds=retry_after,
            )
            response.headers["Retry-After"] = str(retry_after)
            return response
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path == "/docs" and self.settings.app_env != "production":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data:; frame-ancestors 'none'"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; frame-ancestors 'none'"
            )
        response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
        # Browsers only honor HSTS over a secure connection, so it is safe to
        # emit at the origin even when TLS terminates at Vercel/Cloudflare.
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        route_name = getattr(route, "path", "unmatched")
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, route_name, str(response.status_code)).inc()
        HTTP_DURATION.labels(request.method, route_name).observe(elapsed)
        return response


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Trace requests using route templates and an intentionally small attribute allowlist."""

    def __init__(self, app: ASGIApp, tracer: Tracer) -> None:
        super().__init__(app)
        self.tracer = tracer

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        method = _safe_http_method(request.method)
        status_code: int | None = None
        failed = False
        with self.tracer.start_span(
            f"{method} unmatched",
            kind=TraceKind.SERVER,
        ) as span:
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            except BaseException:
                failed = True
                raise
            finally:
                template = _route_template(request)
                span.update_name(f"{method} {template}")
                span.set_attribute("http.request.method", method)
                span.set_attribute("http.route", template)
                if status_code is not None:
                    span.set_attribute("http.response.status_code", status_code)
                if failed or (status_code is not None and status_code >= 400):
                    span.set_error()
