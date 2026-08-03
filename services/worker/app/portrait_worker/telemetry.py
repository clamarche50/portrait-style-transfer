from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from celery import signals
from portrait_api.telemetry import TelemetryRuntime, TraceKind, TraceSpan, create_telemetry

RuntimeFactory = Callable[[str, str | None], TelemetryRuntime]
_SAFE_TASK_NAMES = frozenset(
    {
        "portrait_worker.index_style",
        "portrait_worker.process_transfer_job",
        "portrait_worker.purge_expired",
    }
)
_SAFE_TASK_STATES = frozenset(
    {
        "FAILURE",
        "IGNORED",
        "PENDING",
        "RECEIVED",
        "REJECTED",
        "RETRY",
        "REVOKED",
        "STARTED",
        "SUCCESS",
    }
)


def _safe_task_name(task: object | None) -> str:
    name = getattr(task, "name", None)
    return name if isinstance(name, str) and name in _SAFE_TASK_NAMES else "portrait_worker.other"


def _safe_task_state(state: str | None) -> str:
    normalized = (state or "").upper()
    return normalized.lower() if normalized in _SAFE_TASK_STATES else "unknown"


class CeleryTracing:
    """Manual Celery spans with no task arguments, identifiers, results, or exceptions."""

    def __init__(
        self,
        endpoint: str | None,
        *,
        runtime_factory: RuntimeFactory = create_telemetry,
    ) -> None:
        self.enabled = bool((endpoint or "").strip())
        self._endpoint = endpoint
        self._runtime_factory = runtime_factory
        self._runtime: TelemetryRuntime | None = None
        self._active: dict[str, TraceSpan] = {}
        self._lock = Lock()

    def initialize(self, **_kwargs: object) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._runtime is None:
                self._runtime = self._runtime_factory("portrait-style-worker", self._endpoint)

    def task_prerun(
        self,
        *,
        task_id: str | None = None,
        task: object | None = None,
        **_kwargs: object,
    ) -> None:
        if not self.enabled or task_id is None:
            return
        self.initialize()
        with self._lock:
            if self._runtime is None:
                return
            safe_name = _safe_task_name(task)
            span = self._runtime.tracer.start_span(
                f"celery {safe_name}",
                kind=TraceKind.CONSUMER,
                attributes={
                    "messaging.system": "celery",
                    "celery.task.name": safe_name,
                },
            )
            previous = self._active.pop(task_id, None)
            if previous is not None:
                previous.set_error()
                previous.end()
            self._active[task_id] = span

    def task_postrun(
        self,
        *,
        task_id: str | None = None,
        state: str | None = None,
        **_kwargs: object,
    ) -> None:
        if task_id is None:
            return
        with self._lock:
            span = self._active.pop(task_id, None)
        if span is None:
            return
        safe_state = _safe_task_state(state)
        span.set_attribute("celery.task.state", safe_state)
        if safe_state in {"failure", "rejected", "revoked"}:
            span.set_error()
        span.end()

    def shutdown(self, **_kwargs: object) -> None:
        with self._lock:
            active = list(self._active.values())
            self._active.clear()
            runtime = self._runtime
            self._runtime = None
        for span in active:
            span.set_attribute("celery.task.state", "abandoned")
            span.set_error()
            span.end()
        if runtime is not None:
            runtime.shutdown()


def install_celery_tracing(endpoint: str | None) -> CeleryTracing:
    tracing = CeleryTracing(endpoint)
    signals.worker_process_init.connect(
        tracing.initialize,
        weak=False,
        dispatch_uid="portrait-worker-telemetry-init",
    )
    signals.task_prerun.connect(
        tracing.task_prerun,
        weak=False,
        dispatch_uid="portrait-worker-telemetry-prerun",
    )
    signals.task_postrun.connect(
        tracing.task_postrun,
        weak=False,
        dispatch_uid="portrait-worker-telemetry-postrun",
    )
    signals.worker_process_shutdown.connect(
        tracing.shutdown,
        weak=False,
        dispatch_uid="portrait-worker-telemetry-process-shutdown",
    )
    signals.worker_shutdown.connect(
        tracing.shutdown,
        weak=False,
        dispatch_uid="portrait-worker-telemetry-worker-shutdown",
    )
    return tracing
