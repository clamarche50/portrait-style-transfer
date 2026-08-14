from __future__ import annotations

from portrait_api.middleware import _rate_limit_client_ip
from starlette.requests import Request


def _request(peer: str, headers: list[tuple[bytes, bytes]]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/health/ready",
            "headers": headers,
            "client": (peer, 12345),
            "server": ("api", 8000),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_rate_limit_ip_ignores_attacker_forwarded_for() -> None:
    request = _request(
        "172.19.0.7",
        [(b"x-forwarded-for", b"198.51.100.44")],
    )
    assert _rate_limit_client_ip(request) == "172.19.0.7"


def test_rate_limit_ip_accepts_cloudflare_header_from_private_tunnel_peer() -> None:
    request = _request(
        "172.19.0.7",
        [
            (b"cf-connecting-ip", b"203.0.113.28"),
            (b"cf-ray", b"9abcdef012345678-YYZ"),
            (b"x-forwarded-for", b"192.0.2.99"),
        ],
    )
    assert _rate_limit_client_ip(request) == "203.0.113.28"


def test_rate_limit_ip_rejects_cloudflare_header_from_public_peer() -> None:
    request = _request(
        "8.8.8.8",
        [
            (b"cf-connecting-ip", b"203.0.113.28"),
            (b"cf-ray", b"9abcdef012345678-YYZ"),
        ],
    )
    assert _rate_limit_client_ip(request) == "8.8.8.8"
