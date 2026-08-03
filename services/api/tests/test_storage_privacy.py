from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError
from portrait_api.services.storage import Boto3ObjectStorage


class CompatibleClient:
    def __init__(self, *, public_acl: bool = False, public_policy: bool = False) -> None:
        self.public_acl = public_acl
        self.public_policy = public_policy

    def head_bucket(self, **_kwargs: object) -> None:
        return None

    def put_public_access_block(self, **_kwargs: object) -> None:
        raise ClientError(
            {"Error": {"Code": "MalformedXML", "Message": "unsupported"}},
            "PutPublicAccessBlock",
        )

    def get_bucket_acl(self, **_kwargs: object) -> dict[str, Any]:
        grants: list[dict[str, object]] = []
        if self.public_acl:
            grants.append({"Grantee": {"URI": "http://acs.amazonaws.com/groups/global/AllUsers"}})
        return {"Grants": grants}

    def get_bucket_policy(self, **_kwargs: object) -> dict[str, str]:
        if not self.public_policy:
            raise ClientError(
                {"Error": {"Code": "NoSuchBucketPolicy", "Message": "none"}},
                "GetBucketPolicy",
            )
        return {
            "Policy": (
                '{"Statement":{"Effect":"Allow","Principal":"*",'
                '"Action":"s3:GetObject","Resource":"arn:aws:s3:::bucket/*"}}'
            )
        }


def _storage(client: CompatibleClient) -> Boto3ObjectStorage:
    storage = object.__new__(Boto3ObjectStorage)
    storage.bucket = "bucket"
    storage._custom_endpoint = True
    storage.client = client  # type: ignore[assignment]
    return storage


def test_compatible_storage_verifies_private_acl_and_absent_policy() -> None:
    _storage(CompatibleClient()).ensure_private_bucket()


@pytest.mark.parametrize("mode", ["acl", "policy"])
def test_compatible_storage_fails_closed_for_public_access(mode: str) -> None:
    storage = _storage(CompatibleClient(public_acl=mode == "acl", public_policy=mode == "policy"))

    with pytest.raises(RuntimeError, match="public"):
        storage.ensure_private_bucket()
