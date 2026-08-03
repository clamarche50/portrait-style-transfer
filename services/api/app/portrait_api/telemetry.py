from __future__ import annotations

import json
import queue
import secrets
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from threading import Lock, Thread
from types import TracebackType
from typing import Protocol, Self
from urllib.parse import urlsplit, urlunsplit

AttributeValue = str | int | bool


class TraceKind(IntEnum):
    SERVER = 2
    CONSUMER = 5


@dataclass(frozen=True, slots=True)
class ExportedSpan:
    name: str
    trace_id: str
    span_id: str
    kind: TraceKind
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: Mapping[str, AttributeValue]
    error: bool


class SpanExporter(Protocol):
    def emit(self, span: ExportedSpan) -> None: ...

    def shutdown(self) -> None: ...


class TraceSpan:
    def __init__(
        self,
        name: str,
        kind: TraceKind,
        exporter: SpanExporter | None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.attributes: dict[str, AttributeValue] = dict(attributes or {})
        self.error = False
        self._exporter = exporter
        self._trace_id = secrets.token_hex(16)
        self._span_id = secrets.token_hex(8)
        self._started_at = time.time_ns()
        self._ended = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.set_error()
        self.end()

    def update_name(self, name: str) -> None:
        self.name = name

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self.attributes[key] = value

    def set_error(self) -> None:
        self.error = True

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._exporter is None:
            return
        self._exporter.emit(
            ExportedSpan(
                name=self.name,
                trace_id=self._trace_id,
                span_id=self._span_id,
                kind=self.kind,
                start_time_unix_nano=self._started_at,
                end_time_unix_nano=time.time_ns(),
                attributes=dict(self.attributes),
                error=self.error,
            )
        )


@dataclass(frozen=True, slots=True)
class Tracer:
    exporter: SpanExporter | None

    def start_span(
        self,
        name: str,
        *,
        kind: TraceKind,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> TraceSpan:
        return TraceSpan(name, kind, self.exporter, attributes)


class OtlpHttpJsonExporter:
    """Small OTLP/HTTP JSON batch exporter with no request-path data collection."""

    _stop = object()

    def __init__(self, endpoint: str, service_name: str) -> None:
        self._endpoint = _traces_endpoint(endpoint)
        self._service_name = service_name
        self._queue: queue.Queue[ExportedSpan | object] = queue.Queue(maxsize=2048)
        self._thread: Thread | None = None
        self._lock = Lock()
        self._closed = False

    def emit(self, span: ExportedSpan) -> None:
        with self._lock:
            if self._closed:
                return
            if self._thread is None:
                self._thread = Thread(
                    target=self._run,
                    name="portrait-otlp-exporter",
                    daemon=True,
                )
                self._thread.start()
            try:
                self._queue.put_nowait(span)
            except queue.Full:
                return

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            if thread is None:
                return
            self._queue.put(self._stop)
        thread.join(timeout=6.0)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._stop:
                return
            spans = [item]
            stop_after_batch = False
            deadline = time.monotonic() + 0.2
            while len(spans) < 64:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if next_item is self._stop:
                    stop_after_batch = True
                    break
                spans.append(next_item)
            self._send([span for span in spans if isinstance(span, ExportedSpan)])
            if stop_after_batch:
                return

    def _send(self, spans: Sequence[ExportedSpan]) -> None:
        body = json.dumps(
            _otlp_payload(self._service_name, spans),
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                response.read(0)
        except Exception:
            # Telemetry transport must never affect application availability or expose its URL.
            return


@dataclass(slots=True)
class TelemetryRuntime:
    tracer: Tracer
    exporter: SpanExporter | None
    enabled: bool
    shutdown_called: bool = False

    def shutdown(self) -> None:
        if self.shutdown_called:
            return
        self.shutdown_called = True
        if self.exporter is not None:
            self.exporter.shutdown()


def create_telemetry(
    service_name: str,
    endpoint: str | None,
    *,
    exporter: SpanExporter | None = None,
) -> TelemetryRuntime:
    configured_endpoint = (endpoint or "").strip()
    if not configured_endpoint:
        return TelemetryRuntime(tracer=Tracer(None), exporter=None, enabled=False)
    configured_exporter = exporter or OtlpHttpJsonExporter(configured_endpoint, service_name)
    return TelemetryRuntime(
        tracer=Tracer(configured_exporter),
        exporter=configured_exporter,
        enabled=True,
    )


def _traces_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT must be an HTTP(S) URL")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1/traces"):
        path = f"{path}/v1/traces"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _otlp_payload(service_name: str, spans: Sequence[ExportedSpan]) -> dict[str, object]:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": service_name},
                        },
                        {
                            "key": "service.version",
                            "value": {"stringValue": "0.1.0"},
                        },
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": service_name, "version": "0.1.0"},
                        "spans": [_otlp_span(span) for span in spans],
                    }
                ],
            }
        ]
    }


def _otlp_span(span: ExportedSpan) -> dict[str, object]:
    return {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "kind": int(span.kind),
        "startTimeUnixNano": str(span.start_time_unix_nano),
        "endTimeUnixNano": str(span.end_time_unix_nano),
        "flags": 1,
        "attributes": [
            {"key": key, "value": _otlp_value(value)} for key, value in span.attributes.items()
        ],
        "status": {"code": 2 if span.error else 0},
    }


def _otlp_value(value: AttributeValue) -> dict[str, object]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    return {"stringValue": value}
