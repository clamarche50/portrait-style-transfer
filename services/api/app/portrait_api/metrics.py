from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "portrait_http_requests_total",
    "HTTP requests",
    ("method", "route", "status"),
)
HTTP_DURATION = Histogram(
    "portrait_http_request_duration_seconds",
    "HTTP request duration",
    ("method", "route"),
)
UPLOAD_BYTES = Histogram(
    "portrait_upload_bytes",
    "Normalized upload size",
    buckets=(1024, 10_000, 100_000, 1_000_000, 5_000_000, 15_000_000),
)
JOBS = Counter("portrait_jobs_total", "Jobs by outcome", ("status",))
JOBS_RUNNING = Gauge("portrait_jobs_running", "Jobs currently running")
VALIDATION_ERRORS = Counter(
    "portrait_validation_errors_total",
    "Validation errors",
    ("code",),
)
STORAGE_ERRORS = Counter("portrait_storage_errors_total", "Object storage failures", ("operation",))
EXPIRED_ASSET_LAG = Histogram(
    "portrait_expired_asset_deletion_lag_seconds",
    "Lag between expiry and deletion",
)
