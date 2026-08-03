from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError
from portrait_api.config import Settings


def _contains_public_principal(value: object) -> bool:
    if value == "*":
        return True
    if isinstance(value, dict):
        return any(_contains_public_principal(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_public_principal(item) for item in value)
    return False


class ObjectStorage(Protocol):
    def ping(self) -> bool: ...

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...

    def delete_prefix(self, prefix: str) -> int: ...

    def signed_download_url(self, key: str, expires_seconds: int) -> str: ...


class Boto3ObjectStorage:
    def __init__(self, settings: Settings) -> None:
        addressing_style = "path" if settings.s3_force_path_style else "virtual"
        self.bucket = settings.s3_bucket
        self._custom_endpoint = settings.s3_endpoint_url is not None
        self._put_encryption: dict[str, str] = {}
        if settings.s3_server_side_encryption != "none":
            self._put_encryption["ServerSideEncryption"] = settings.s3_server_side_encryption
        if settings.s3_server_side_encryption == "aws:kms" and settings.s3_kms_key_id:
            self._put_encryption["SSEKMSKeyId"] = settings.s3_kms_key_id
        self.client: BaseClient = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id.get_secret_value(),
            aws_secret_access_key=settings.s3_secret_access_key.get_secret_value(),
            config=Config(signature_version="s3v4", s3={"addressing_style": addressing_style}),
        )

    def ensure_private_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            self.client.create_bucket(Bucket=self.bucket)
        try:
            self.client.put_public_access_block(
                Bucket=self.bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if not self._custom_endpoint or code not in {
                "MalformedXML",
                "NotImplemented",
                "UnsupportedOperation",
            }:
                raise
            self._verify_private_compatible_bucket()

    def _verify_private_compatible_bucket(self) -> None:
        acl = self.client.get_bucket_acl(Bucket=self.bucket)
        public_acl_uris = {
            "http://acs.amazonaws.com/groups/global/AllUsers",
            "http://acs.amazonaws.com/groups/global/AuthenticatedUsers",
        }
        for grant in acl.get("Grants", []):
            grantee = grant.get("Grantee", {})
            if isinstance(grantee, dict) and grantee.get("URI") in public_acl_uris:
                raise RuntimeError("Object-storage bucket ACL is public")

        try:
            response = self.client.get_bucket_policy(Bucket=self.bucket)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchBucketPolicy", "NoSuchPolicy", "404", "NotFound"}:
                return
            raise
        policy = json.loads(str(response.get("Policy", "{}")))
        statements = policy.get("Statement", []) if isinstance(policy, dict) else []
        if isinstance(statements, dict):
            statements = [statements]
        for statement in statements:
            if (
                isinstance(statement, dict)
                and statement.get("Effect") == "Allow"
                and _contains_public_principal(statement.get("Principal"))
            ):
                raise RuntimeError("Object-storage bucket policy allows public access")

    def ping(self) -> bool:
        self.client.head_bucket(Bucket=self.bucket)
        return True

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={"privacy": "private"},
            **self._put_encryption,
        )

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return bytes(response["Body"].read())

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> int:
        deleted = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if objects:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
                deleted += len(objects)
        return deleted

    def signed_download_url(self, key: str, expires_seconds: int) -> str:
        return str(
            self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )
        )


@dataclass(slots=True)
class _MemoryObject:
    data: bytes
    content_type: str


class MemoryObjectStorage:
    """Thread-safe test adapter; never selected by production settings."""

    def __init__(self) -> None:
        self.objects: dict[str, _MemoryObject] = {}
        self._lock = Lock()

    def ping(self) -> bool:
        return True

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        with self._lock:
            self.objects[key] = _MemoryObject(data=bytes(data), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        with self._lock:
            return self.objects[key].data

    def delete(self, key: str) -> None:
        with self._lock:
            self.objects.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [key for key in self.objects if key.startswith(prefix)]
            for key in keys:
                del self.objects[key]
            return len(keys)

    def signed_download_url(self, key: str, expires_seconds: int) -> str:
        if key not in self.objects:
            raise KeyError(key)
        return f"https://objects.invalid/{key}?expires={expires_seconds}"
