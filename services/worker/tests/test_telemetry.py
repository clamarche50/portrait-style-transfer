from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from portrait_api.telemetry import ExportedSpan, TelemetryRuntime, create_telemetry
from portrait_worker.telemetry import CeleryTracing


@dataclass(slots=True)
class FakeTask:
    name: str


@dataclass(slots=True)
class MemorySpanExporter:
    spans: list[ExportedSpan] = field(default_factory=list)
    shutdown_called: bool = False

    def emit(self, span: ExportedSpan) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        self.shutdown_called = True


def _memory_runtime() -> tuple[TelemetryRuntime, MemorySpanExporter]:
    exporter = MemorySpanExporter()
    runtime = create_telemetry(
        "portrait-style-worker",
        "http://collector.invalid:4318",
        exporter=exporter,
    )
    return runtime, exporter


def test_blank_endpoint_does_not_create_a_worker_runtime() -> None:
    calls = 0

    def factory(_service_name: str, _endpoint: str | None) -> TelemetryRuntime:
        nonlocal calls
        calls += 1
        runtime, _exporter = _memory_runtime()
        return runtime

    tracing = CeleryTracing("  ", runtime_factory=factory)
    tracing.initialize()
    tracing.task_prerun(task_id=str(uuid.uuid4()), task=FakeTask("portrait_worker.index_style"))
    tracing.shutdown()

    assert tracing.enabled is False
    assert calls == 0


def test_task_span_excludes_arguments_ids_results_and_exceptions() -> None:
    runtime, exporter = _memory_runtime()
    tracing = CeleryTracing(
        "http://collector:4318",
        runtime_factory=lambda _service_name, _endpoint: runtime,
    )
    private_task_id = str(uuid.uuid4())
    secrets = (
        private_task_id,
        str(uuid.uuid4()),
        "uploads/private/headshot.jpg",
        "https://storage.invalid/private?X-Amz-Signature=secret",
        "private-result",
        "private-exception-message",
    )

    tracing.task_prerun(
        task_id=private_task_id,
        task=FakeTask("portrait_worker.process_transfer_job"),
        args=(secrets[1], secrets[2]),
        kwargs={"download_url": secrets[3]},
    )
    tracing.task_postrun(
        task_id=private_task_id,
        state="FAILURE",
        retval=secrets[4],
        exception=RuntimeError(secrets[5]),
    )
    tracing.shutdown()

    assert runtime.shutdown_called is True
    assert exporter.shutdown_called is True
    assert len(exporter.spans) == 1
    span = exporter.spans[0]
    assert span.name == "celery portrait_worker.process_transfer_job"
    assert span.attributes == {
        "messaging.system": "celery",
        "celery.task.name": "portrait_worker.process_transfer_job",
        "celery.task.state": "failure",
    }
    assert span.error is True
    serialized_span = repr(span)
    for secret in secrets:
        assert secret not in serialized_span


def test_unknown_task_name_is_redacted() -> None:
    runtime, exporter = _memory_runtime()
    tracing = CeleryTracing(
        "http://collector:4318",
        runtime_factory=lambda _service_name, _endpoint: runtime,
    )
    private_name = f"portrait_worker.dynamic.{uuid.uuid4()}"
    task_id = str(uuid.uuid4())

    tracing.task_prerun(task_id=task_id, task=FakeTask(private_name))
    tracing.task_postrun(task_id=task_id, state="SUCCESS")
    tracing.shutdown()

    span = exporter.spans[0]
    assert span.name == "celery portrait_worker.other"
    assert span.attributes["celery.task.name"] == "portrait_worker.other"
    assert private_name not in repr(span)
