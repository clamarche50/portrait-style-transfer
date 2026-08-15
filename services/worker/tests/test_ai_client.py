from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from portrait_worker import ai_client
from portrait_worker.ai_client import AIEngineClient, AIEngineError
from pydantic import SecretStr


def _png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, "PNG")
    return output.getvalue()


def _jpeg(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, "JPEG")
    return output.getvalue()


def _diagnostics_header(value: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode("utf-8"))
    return encoded.decode("ascii").rstrip("=")


def test_transfer_uses_explicit_content_and_style_multipart_roles() -> None:
    content = _png((12, 34, 56))
    style = _png((210, 180, 150))
    api_token = "worker-test-secret-token"
    observed: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        observed["body"] = body
        assert request.url.path == "/v1/transfer"
        assert request.headers["Authorization"] == f"Bearer {api_token}"
        assert b'name="content"; filename="content.png"' in body
        assert b'name="style"; filename="style.png"' in body
        assert content in body
        assert style in body
        assert b'"random_seed":17' in body
        return httpx.Response(
            200,
            content=content,
            headers={
                "Content-Type": "image/png",
                "X-Portrait-Engine": "ai_instantstyle_v1",
                "X-Portrait-Diagnostics": _diagnostics_header(
                    {"model": "InstantStyle", "seed": 17}
                ),
            },
        )

    client = AIEngineClient(
        base_url="http://engine.test",
        api_token=api_token,
        transport=httpx.MockTransport(handler),
    )
    response = client.transfer(
        content=content,
        style=style,
        settings={"random_seed": 17},
    )

    assert observed["body"].find(content) < observed["body"].find(style)
    assert response.image_png == content
    assert response.engine_id == "ai_instantstyle_v1"
    assert response.diagnostics == {"model": "InstantStyle", "seed": 17}
    assert api_token not in repr(client)
    with pytest.raises(TypeError):
        client.transfer(content, style, {})  # type: ignore[misc]


def test_transfer_maps_structured_engine_error_without_leaking_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "detail": {
                    "code": "MODEL_NOT_READY",
                    "message": "The portrait model is warming up.",
                    "retryable": True,
                }
            },
        )

    client = AIEngineClient(
        base_url="http://engine.test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AIEngineError) as failure:
        client.transfer(content=_png((1, 2, 3)), style=_png((4, 5, 6)), settings={})

    assert failure.value.code == "MODEL_NOT_READY"
    assert failure.value.safe_message == "The portrait model is warming up."
    assert failure.value.retryable is True


def test_transfer_preserves_bounded_jpeg_upload_representation() -> None:
    content = _jpeg((1, 2, 3))

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        assert b'filename="content.jpg"' in body
        assert b"Content-Type: image/jpeg" in body
        assert content in body
        return httpx.Response(
            200,
            content=_png((1, 2, 3)),
            headers={"Content-Type": "image/png", "X-Portrait-Engine": "ai_instantstyle_v1"},
        )

    client = AIEngineClient(base_url="http://engine.test", transport=httpx.MockTransport(handler))
    response = client.transfer(content=content, style=_png((4, 5, 6)), settings={})
    assert response.image_png.startswith(b"\x89PNG")


@pytest.mark.parametrize(
    ("headers", "expected_message"),
    [
        (
            {"Content-Type": "image/png", "X-Portrait-Engine": "wrong-engine"},
            "unexpected engine identifier",
        ),
        (
            {
                "Content-Type": "image/png",
                "X-Portrait-Engine": "ai_instantstyle_v1",
                "X-Portrait-Diagnostics": "not+valid+base64",
            },
            "invalid diagnostics",
        ),
    ],
)
def test_transfer_rejects_invalid_success_contract(
    headers: dict[str, str], expected_message: str
) -> None:
    client = AIEngineClient(
        base_url="http://engine.test",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=_png((1, 2, 3)), headers=headers)
        ),
    )
    with pytest.raises(AIEngineError, match=expected_message) as failure:
        client.transfer(content=_png((1, 1, 1)), style=_png((2, 2, 2)), settings={})
    assert failure.value.code == "AI_ENGINE_INVALID_RESPONSE"
    assert failure.value.retryable is False


def test_transfer_classifies_transport_timeout_as_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    client = AIEngineClient(
        base_url="http://engine.test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AIEngineError) as failure:
        client.transfer(content=_png((1, 1, 1)), style=_png((2, 2, 2)), settings={})
    assert failure.value.code == "AI_ENGINE_TIMEOUT"
    assert failure.value.retryable is True


def test_configured_client_uses_8010_and_keeps_token_out_of_cache_key_and_repr() -> None:
    token = "cache-key-must-not-contain-this-token"
    settings = SimpleNamespace(
        ai_engine_url="http://ai-engine:8010",
        ai_engine_request_timeout_seconds=600,
        ai_engine_api_token=SecretStr(token),
    )
    client = ai_client.get_ai_engine_client(settings)
    try:
        assert "8010" in repr(client)
        assert token not in repr(client)
        assert all(token not in repr(key) for key in ai_client._CLIENTS)
    finally:
        client.close()
        ai_client._CLIENTS.clear()
