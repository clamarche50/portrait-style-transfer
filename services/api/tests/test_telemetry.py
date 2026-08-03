from __future__ import annotations

import json
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import pytest
from fastapi.testclient import TestClient
from portrait_api.config import Settings
from portrait_api.db import build_engine
from portrait_api.main import ServiceOverrides, create_app
from portrait_api.services.queue import MemoryTaskQueue
from portrait_api.services.redis_gateway import MemoryProgressStore
from portrait_api.services.storage import MemoryObjectStorage
from portrait_api.telemetry import (
    ExportedSpan,
    SpanExporter,
    TelemetryRuntime,
    TraceKind,
    create_telemetry,
)


@dataclass(slots=True)
class MemorySpanExporter:
    spans: list[ExportedSpan] = field(default_factory=list)
    shutdown_called: bool = False

    def emit(self, span: ExportedSpan) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeHttpResponse:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self, _amount: int) -> bytes:
        return b""


def _memory_runtime() -> tuple[TelemetryRuntime, MemorySpanExporter]:
    exporter = MemorySpanExporter()
    runtime = create_telemetry(
        "portrait-style-api",
        "http://collector.invalid:4318",
        exporter=exporter,
    )
    return runtime, exporter


@contextmanager
def _client(runtime: TelemetryRuntime) -> Iterator[TestClient]:
    settings = Settings(
        app_env="test",
        database_url="sqlite+pysqlite://",
        require_models_for_readiness=False,
        initialize_storage_on_startup=False,
        auto_create_schema=True,
        cookie_secure_override=False,
        session_secret="telemetry-test-secret-with-sufficient-entropy",
        cors_origins=["http://testserver"],
    )
    app = create_app(
        settings,
        overrides=ServiceOverrides(
            engine=build_engine(settings.database_url),
            storage=MemoryObjectStorage(),
            progress_store=MemoryProgressStore(),
            task_queue=MemoryTaskQueue(),
            telemetry=runtime,
        ),
    )
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("endpoint", [None, "", "   "])
def test_blank_endpoint_keeps_tracing_disabled(endpoint: str | None) -> None:
    exporter: SpanExporter = MemorySpanExporter()
    runtime = create_telemetry("test-api", endpoint, exporter=exporter)

    assert runtime.enabled is False
    assert runtime.exporter is None
    runtime.shutdown()
    assert runtime.shutdown_called is True
    assert isinstance(exporter, MemorySpanExporter)
    assert exporter.shutdown_called is False


def test_request_span_uses_only_safe_route_metadata_and_shuts_down() -> None:
    runtime, exporter = _memory_runtime()
    private_job_id = str(uuid.uuid4())
    secrets = (
        private_job_id,
        "private-headshot.jpg",
        "private-object-key",
        "signed-url-secret",
        "private-cookie-value",
        "private-request-body",
    )

    with _client(runtime) as client:
        response = client.get(
            f"/api/v1/jobs/{private_job_id}",
            params={
                "filename": secrets[1],
                "object_key": secrets[2],
                "X-Amz-Signature": secrets[3],
                "body": secrets[5],
            },
            headers={"Cookie": f"unrelated={secrets[4]}"},
        )

    assert response.status_code == 404
    assert runtime.shutdown_called is True
    assert exporter.shutdown_called is True
    assert len(exporter.spans) == 1
    span = exporter.spans[0]
    assert span.name == "GET /api/v1/jobs/{job_id}"
    assert span.attributes == {
        "http.request.method": "GET",
        "http.route": "/api/v1/jobs/{job_id}",
        "http.response.status_code": 404,
    }
    assert span.error is True
    serialized_span = repr(span)
    for secret in secrets:
        assert secret not in serialized_span


def test_otlp_http_json_payload_and_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[urllib.request.Request, float]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHttpResponse:
        requests.append((request, timeout))
        return FakeHttpResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    runtime = create_telemetry("portrait-style-api", "http://collector.invalid:4318")
    with runtime.tracer.start_span("GET /api/v1/health/live", kind=TraceKind.SERVER) as span:
        span.set_attribute("http.request.method", "GET")
        span.set_attribute("http.route", "/api/v1/health/live")
        span.set_attribute("http.response.status_code", 200)
    runtime.shutdown()

    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == "http://collector.invalid:4318/v1/traces"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 5.0
    assert request.data is not None
    payload = json.loads(request.data)
    resource_span = payload["resourceSpans"][0]
    resource_attributes = resource_span["resource"]["attributes"]
    assert resource_attributes[0]["value"]["stringValue"] == "portrait-style-api"
    exported_span = resource_span["scopeSpans"][0]["spans"][0]
    assert len(exported_span["traceId"]) == 32
    assert len(exported_span["spanId"]) == 16
    assert exported_span["kind"] == 2
    assert exported_span["flags"] == 1
    assert exported_span["status"] == {"code": 0}
    assert exported_span["attributes"] == [
        {"key": "http.request.method", "value": {"stringValue": "GET"}},
        {"key": "http.route", "value": {"stringValue": "/api/v1/health/live"}},
        {"key": "http.response.status_code", "value": {"intValue": "200"}},
    ]
