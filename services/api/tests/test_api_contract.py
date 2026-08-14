from __future__ import annotations

import asyncio
import io
import uuid
from datetime import UTC, datetime

from conftest import ApiHarness, png_bytes
from PIL import Image, PngImagePlugin
from portrait_api.models import ArtifactKind, Asset, AssetKind, Job, JobStatus, StyleExample
from portrait_api.repositories import AssetRepository, JobRepository
from sqlalchemy.orm import Session


def _upload(api: ApiHarness, kind: str, payload: bytes, *, name: str = "portrait.png") -> dict:
    response = api.client.post(
        "/api/v1/assets/upload",
        data={"kind": kind},
        files={"file": (name, payload, "image/png")},
        headers=api.unsafe_headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health_error_envelope_security_headers_and_csrf(api: ApiHarness) -> None:
    ready = api.client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {
        "database": "ok",
        "redis": "ok",
        "object_storage": "ok",
        "models": "not_required",
    }
    assert ready.headers["x-content-type-options"] == "nosniff"
    assert ready.headers["x-frame-options"] == "DENY"

    csrf_failure = api.client.post(
        "/api/v1/styles",
        json={"name": "Private", "rights_confirmed": True},
    )
    assert csrf_failure.status_code == 403
    assert csrf_failure.json()["error"]["code"] == "CSRF_FAILED"
    assert csrf_failure.headers["x-content-type-options"] == "nosniff"
    assert csrf_failure.headers["x-request-id"]
    recovered = api.client.post(
        "/api/v1/styles",
        json={"name": "Recovered", "rights_confirmed": True},
        headers={"X-CSRF-Token": csrf_failure.headers["X-CSRF-Token"]},
    )
    assert recovered.status_code == 201, recovered.text

    invalid = api.client.get("/api/v1/assets/not-a-uuid")
    assert invalid.status_code == 422
    assert set(invalid.json()["error"]) == {"code", "message", "details", "request_id"}


def test_upload_normalizes_metadata_and_enforces_anonymous_ownership(api: ApiHarness) -> None:
    image = Image.new("RGB", (80, 96), (20, 30, 40))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private-note", "must-not-survive")
    source = io.BytesIO()
    image.save(source, "PNG", pnginfo=metadata)

    asset = _upload(api, "INPUT", source.getvalue(), name="sensitive-person-name.png")
    assert "object_key" not in asset
    assert asset["metadata"] == {"normalized": True, "source_format": "PNG"}
    stored = next(iter(api.storage.objects.values())).data
    with Image.open(io.BytesIO(stored)) as normalized:
        assert "private-note" not in normalized.info

    protected_download = api.client.get(f"/api/v1/assets/{asset['id']}/content?download=true")
    assert protected_download.status_code == 403

    signed = api.client.post(
        f"/api/v1/assets/{asset['id']}/download-url",
        headers=api.unsafe_headers,
    )
    assert signed.status_code == 200
    assert signed.json()["url"] == f"/api/v1/assets/{asset['id']}/content?download=true"
    assert signed.json()["expires_at"]
    assert "pst_download=" in signed.headers["set-cookie"]
    assert "HttpOnly" in signed.headers["set-cookie"]
    assert "SameSite=strict" in signed.headers["set-cookie"]
    assert "token=" not in signed.json()["url"]

    downloaded = api.client.get(signed.json()["url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-disposition"].startswith("attachment")
    replay = api.client.get(signed.json()["url"])
    assert replay.status_code == 403

    content = api.client.get(f"/api/v1/assets/{asset['id']}/content")
    assert content.status_code == 200
    assert content.headers["content-type"] == "image/png"
    assert content.content == stored

    api.refresh_session()
    hidden = api.client.get(f"/api/v1/assets/{asset['id']}")
    assert hidden.status_code == 404
    assert hidden.json()["error"]["code"] == "ASSET_NOT_FOUND"


def test_reference_job_lifecycle_sse_cancel_and_download(
    api: ApiHarness, portrait_png: bytes
) -> None:
    input_asset = _upload(api, "INPUT", portrait_png)
    reference_asset = _upload(api, "REFERENCE", png_bytes((180, 140, 110)))
    response = api.client.post(
        "/api/v1/jobs",
        json={
            "input_asset_id": input_asset["id"],
            "reference_asset_id": reference_asset["id"],
            "settings": {
                "algorithm_profile": "ai_dgpst_v1",
                "style_strength": 0.8,
                "structure_strength": 0.95,
                "inference_steps": 24,
                "random_seed": 7,
            },
        },
        headers=api.unsafe_headers,
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["status"] == "QUEUED"
    assert job["algorithm_profile"] == "ai_dgpst_v1"
    assert job["settings"] == {
        "algorithm_profile": "ai_dgpst_v1",
        "style_strength": 0.8,
        "structure_strength": 0.95,
        "inference_steps": 24,
        "random_seed": 7,
        "background_mode": "KEEP",
        "background_color": None,
        "output_format": "PNG",
        "jpeg_quality": 95,
    }
    assert api.queue.transfers == [(job["id"], False)]

    cancelled = api.client.post(f"/api/v1/jobs/{job['id']}/cancel", headers=api.unsafe_headers)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"
    assert job["id"] in api.progress.cancelled

    events = api.client.get(f"/api/v1/jobs/{job['id']}/events")
    assert events.status_code == 200
    assert events.text.startswith("data: ")
    assert '"status":"CANCELLED"' in events.text
    assert '"input_preview_url":"/api/v1/assets/' in events.text

    no_output = api.client.post(
        f"/api/v1/jobs/{job['id']}/download-url", headers=api.unsafe_headers
    )
    assert no_output.status_code == 409


def test_job_settings_reject_classical_engine_controls(
    api: ApiHarness, portrait_png: bytes
) -> None:
    input_asset = _upload(api, "INPUT", portrait_png)
    reference_asset = _upload(api, "REFERENCE", png_bytes((180, 140, 110)))
    response = api.client.post(
        "/api/v1/jobs",
        json={
            "input_asset_id": input_asset["id"],
            "reference_asset_id": reference_asset["id"],
            "settings": {
                "algorithm_profile": "paper_exact",
                "transfer_strength": 1.0,
                "dense_alignment": True,
            },
        },
        headers=api.unsafe_headers,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_openapi_exposes_only_ai_native_transfer_controls(api: ApiHarness) -> None:
    settings_schema = api.client.get("/openapi.json").json()["components"]["schemas"][
        "TransferSettingsRequest"
    ]
    properties = settings_schema["properties"]

    assert set(properties) == {
        "algorithm_profile",
        "style_strength",
        "structure_strength",
        "inference_steps",
        "random_seed",
        "background_mode",
        "background_color",
        "output_format",
        "jpeg_quality",
    }
    assert properties["algorithm_profile"]["default"] == "ai_dgpst_v1"
    assert properties["style_strength"]["default"] == 0.75
    assert properties["structure_strength"]["default"] == 0.9
    assert properties["inference_steps"] == {
        "default": 30,
        "maximum": 50,
        "minimum": 10,
        "title": "Inference Steps",
        "type": "integer",
    }


def test_style_crud_index_and_rank_use_real_feature_code(api: ApiHarness) -> None:
    style_asset = _upload(api, "STYLE_EXAMPLE", png_bytes((210, 170, 125)))
    input_asset = _upload(api, "INPUT", png_bytes((90, 120, 165)))
    created = api.client.post(
        "/api/v1/styles",
        json={
            "name": "Warm studio",
            "description": "Private synthetic test style",
            "rights_confirmed": True,
            "is_public": False,
        },
        headers=api.unsafe_headers,
    )
    assert created.status_code == 201, created.text
    style_id = created.json()["id"]

    added = api.client.post(
        f"/api/v1/styles/{style_id}/examples",
        json={"asset_id": style_asset["id"]},
        headers=api.unsafe_headers,
    )
    assert added.status_code == 201, added.text
    assert added.json()["indexed"] is True
    assert added.json()["quality"]["full_ingestion"] == "QUEUED"
    assert api.queue.style_indexes == [style_id]
    assert any(key.startswith(f"styles/{style_id}/features/") for key in api.storage.objects)

    ranked = api.client.post(
        f"/api/v1/styles/{style_id}/rank",
        json={"input_asset_id": input_asset["id"], "limit": 3},
        headers=api.unsafe_headers,
    )
    assert ranked.status_code == 200, ranked.text
    assert ranked.json()["results"][0]["example_id"] == added.json()["id"]
    ranking_diagnostics = ranked.json()["results"][0]["diagnostics"]
    assert ranking_diagnostics["metric"] == "weighted_compatibility_v1"
    assert ranking_diagnostics["analysis_mode"] == "lightweight_image_statistics"
    assert set(ranking_diagnostics["components"]) == {
        "local_energy_ncc",
        "pose_similarity",
        "landmark_shape_similarity",
        "photometric_compatibility",
        "mask_quality",
    }

    style_job = api.client.post(
        "/api/v1/jobs",
        json={
            "input_asset_id": input_asset["id"],
            "style_id": style_id,
            "settings": {"algorithm_profile": "ai_dgpst_v1"},
        },
        headers=api.unsafe_headers,
    )
    assert style_job.status_code == 202, style_job.text


def test_public_style_preview_is_cross_session_read_only(api: ApiHarness) -> None:
    public_asset = _upload(api, "STYLE_EXAMPLE", png_bytes((215, 175, 130)))
    private_asset = _upload(api, "STYLE_EXAMPLE", png_bytes((80, 95, 110)))
    public_style = api.client.post(
        "/api/v1/styles",
        json={"name": "Public", "rights_confirmed": True, "is_public": True},
        headers=api.unsafe_headers,
    ).json()
    private_style = api.client.post(
        "/api/v1/styles",
        json={"name": "Private", "rights_confirmed": True, "is_public": False},
        headers=api.unsafe_headers,
    ).json()
    for style, asset in ((public_style, public_asset), (private_style, private_asset)):
        added = api.client.post(
            f"/api/v1/styles/{style['id']}/examples",
            json={"asset_id": asset["id"]},
            headers=api.unsafe_headers,
        )
        assert added.status_code == 201, added.text

    api.refresh_session()
    visible = api.client.get(f"/api/v1/styles/{public_style['id']}")
    assert visible.status_code == 200
    preview_url = visible.json()["preview_url"]
    assert preview_url == f"/api/v1/assets/{public_asset['id']}/content"
    assert api.client.get(preview_url).status_code == 200
    assert api.client.get(f"/api/v1/assets/{private_asset['id']}/content").status_code == 404
    assert api.client.get(f"/api/v1/styles/{private_style['id']}").status_code == 404


def test_deleting_style_removes_unreferenced_example_bytes_and_records(api: ApiHarness) -> None:
    style_asset = _upload(api, "STYLE_EXAMPLE", png_bytes((200, 160, 120)))
    input_asset = _upload(api, "INPUT", png_bytes((90, 120, 165)))
    style = api.client.post(
        "/api/v1/styles",
        json={"name": "Disposable", "rights_confirmed": True},
        headers=api.unsafe_headers,
    ).json()
    example = api.client.post(
        f"/api/v1/styles/{style['id']}/examples",
        json={"asset_id": style_asset["id"]},
        headers=api.unsafe_headers,
    ).json()
    created_job = api.client.post(
        "/api/v1/jobs",
        json={"input_asset_id": input_asset["id"], "style_id": style["id"]},
        headers=api.unsafe_headers,
    )
    assert created_job.status_code == 202
    blocked = api.client.delete(f"/api/v1/styles/{style['id']}", headers=api.unsafe_headers)
    assert blocked.status_code == 409
    cancelled = api.client.post(
        f"/api/v1/jobs/{created_job.json()['id']}/cancel", headers=api.unsafe_headers
    )
    assert cancelled.status_code == 200

    with Session(api.engine) as db:
        asset = db.get(Asset, uuid.UUID(style_asset["id"]))
        stored_key = asset.object_key if asset is not None else ""
        style_example = db.get(StyleExample, uuid.UUID(example["id"]))
        feature_key = style_example.feature_object_key if style_example is not None else ""
    derived_key = f"styles/{style['id']}/examples/{example['id']}/head-mask.npy"
    api.storage.put_bytes(derived_key, b"private-derived-data", "application/octet-stream")
    deleted = api.client.delete(f"/api/v1/styles/{style['id']}", headers=api.unsafe_headers)
    assert deleted.status_code == 204, deleted.text
    assert stored_key not in api.storage.objects
    assert feature_key not in api.storage.objects
    assert derived_key not in api.storage.objects
    with Session(api.engine) as db:
        asset = db.get(Asset, uuid.UUID(style_asset["id"]))
        assert asset is not None and asset.deleted_at is not None
        assert db.get(StyleExample, uuid.UUID(example["id"])) is None


def test_deleting_one_example_keeps_shared_asset_until_last_reference(api: ApiHarness) -> None:
    style_asset = _upload(api, "STYLE_EXAMPLE", png_bytes((170, 150, 130)))
    style_ids: list[str] = []
    example_ids: list[str] = []
    for name in ("Shared A", "Shared B"):
        style = api.client.post(
            "/api/v1/styles",
            json={"name": name, "rights_confirmed": True},
            headers=api.unsafe_headers,
        ).json()
        example = api.client.post(
            f"/api/v1/styles/{style['id']}/examples",
            json={"asset_id": style_asset["id"]},
            headers=api.unsafe_headers,
        ).json()
        style_ids.append(style["id"])
        example_ids.append(example["id"])
    with Session(api.engine) as db:
        asset = db.get(Asset, uuid.UUID(style_asset["id"]))
        assert asset is not None
        stored_key = asset.object_key

    first = api.client.delete(
        f"/api/v1/styles/{style_ids[0]}/examples/{example_ids[0]}",
        headers=api.unsafe_headers,
    )
    assert first.status_code == 204
    assert stored_key in api.storage.objects
    assert api.client.get(f"/api/v1/assets/{style_asset['id']}/content").status_code == 200

    second = api.client.delete(
        f"/api/v1/styles/{style_ids[1]}/examples/{example_ids[1]}",
        headers=api.unsafe_headers,
    )
    assert second.status_code == 204
    assert stored_key not in api.storage.objects
    with Session(api.engine) as db:
        asset = db.get(Asset, uuid.UUID(style_asset["id"]))
        assert asset is not None and asset.deleted_at is not None


def test_ai_background_correction_forces_full_rerun_and_clears_private_cache(
    api: ApiHarness,
) -> None:
    input_asset = _upload(api, "INPUT", png_bytes())
    reference_asset = _upload(api, "REFERENCE", png_bytes((170, 150, 120)))
    created = api.client.post(
        "/api/v1/jobs",
        json={
            "input_asset_id": input_asset["id"],
            "reference_asset_id": reference_asset["id"],
        },
        headers=api.unsafe_headers,
    )
    assert created.status_code == 202
    job_id = uuid.UUID(created.json()["id"])
    affine_key = f"jobs/{job_id}/cache/affine_alignment/affine.npy"
    gain_key = f"jobs/{job_id}/cache/multiscale_transfer/gain.npy"
    api.storage.put_bytes(affine_key, b"early", "application/octet-stream")
    api.storage.put_bytes(gain_key, b"late", "application/octet-stream")

    with Session(api.engine) as db, db.begin():
        repository = JobRepository(db)
        job = repository.get_for_worker(job_id, for_update=True)
        assert job is not None
        output_key = f"outputs/{job_id}/attempt-1.png"
        output_bytes = png_bytes((100, 110, 120))
        api.storage.put_bytes(output_key, output_bytes, "image/png")
        output = AssetRepository(db).create(
            session_id=job.session_id,
            kind=AssetKind.OUTPUT,
            object_key=output_key,
            mime_type="image/png",
            width=96,
            height=128,
            byte_size=len(output_bytes),
            sha256="0" * 64,
            metadata={"job_id": str(job.id)},
            expires_at=job.expires_at,
        )
        repository.add_artifact(job.id, output.id, ArtifactKind.OUTPUT)
        repository.mark_succeeded(
            job,
            {
                "summary": {"profile": "ai_dgpst_v1"},
                "private_cache_manifest": {
                    "affine": {"key": affine_key, "stage": "AFFINE_ALIGNMENT"},
                    "gain": {"key": gain_key, "stage": "MULTISCALE_TRANSFER"},
                },
            },
        )

    diagnostics = api.client.get(f"/api/v1/jobs/{job_id}/diagnostics")
    assert diagnostics.status_code == 200
    assert "private_cache_manifest" not in diagnostics.json()["diagnostics"]
    assert affine_key not in diagnostics.text

    corrected = api.client.post(
        f"/api/v1/jobs/{job_id}/corrections",
        json={
            "corrections": [
                {
                    "type": "background",
                    "mode": "BLUR",
                    "color": None,
                }
            ]
        },
        headers=api.unsafe_headers,
    )
    assert corrected.status_code == 200, corrected.text
    assert affine_key not in api.storage.objects
    assert gain_key not in api.storage.objects

    rerun = api.client.post(f"/api/v1/jobs/{job_id}/rerun", headers=api.unsafe_headers)
    assert rerun.status_code == 202, rerun.text
    with Session(api.engine) as db:
        job = db.get(Job, job_id)
        assert job is not None
        assert job.status == JobStatus.QUEUED
        assert job.diagnostics["resume"] == {
            "requested_stage": "VALIDATING",
            "cache_reuse": False,
        }


def test_ai_job_rejects_classical_corrections(api: ApiHarness) -> None:
    input_asset = _upload(api, "INPUT", png_bytes())
    reference_asset = _upload(api, "REFERENCE", png_bytes((170, 150, 120)))
    created = api.client.post(
        "/api/v1/jobs",
        json={"input_asset_id": input_asset["id"], "reference_asset_id": reference_asset["id"]},
        headers=api.unsafe_headers,
    )
    assert created.status_code == 202
    job_id = uuid.UUID(created.json()["id"])
    with Session(api.engine) as db, db.begin():
        job = JobRepository(db).get_for_worker(job_id, for_update=True)
        assert job is not None
        JobRepository(db).mark_failed(job, code="SYNTHETIC", safe_message="Synthetic failure")

    rejected = api.client.post(
        f"/api/v1/jobs/{job_id}/corrections",
        json={
            "corrections": [
                {
                    "type": "mask",
                    "operation": "ADD",
                    "radius": 0.02,
                    "points": [[0.4, 0.4], [0.45, 0.45]],
                }
            ]
        },
        headers=api.unsafe_headers,
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "AI_CORRECTION_UNSUPPORTED"
    with Session(api.engine) as db:
        job = db.get(Job, job_id)
        assert job is not None
        assert not job.corrections


def test_progress_store_terminal_snapshot_is_sse_safe(api: ApiHarness, portrait_png: bytes) -> None:
    input_asset = _upload(api, "INPUT", portrait_png)
    reference_asset = _upload(api, "REFERENCE", portrait_png)
    created = api.client.post(
        "/api/v1/jobs",
        json={"input_asset_id": input_asset["id"], "reference_asset_id": reference_asset["id"]},
        headers=api.unsafe_headers,
    )
    job_id = created.json()["id"]
    with Session(api.engine) as db, db.begin():
        job = JobRepository(db).get_for_worker(uuid.UUID(job_id), for_update=True)
        assert job is not None
        JobRepository(db).mark_failed(job, code="SYNTHETIC", safe_message="Synthetic failure")
    asyncio.run(
        api.progress.set_progress(
            job_id,
            {
                "job_id": job_id,
                "status": "FAILED",
                "stage": "VALIDATING",
                "progress": 1,
                "message": "Synthetic failure",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
    )
    events = api.client.get(f"/api/v1/jobs/{job_id}/events")
    assert events.status_code == 200
    assert "Synthetic failure" in events.text
