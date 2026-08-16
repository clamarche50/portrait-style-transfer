#!/usr/bin/env python3
"""Provision, manifest, and verify the local InstantStyle SDXL inference artifact set.

The portrait engine uses six pinned, license-reviewed model sets:

* stabilityai/stable-diffusion-xl-base-1.0 (fp16) - base diffusion model
* h94/IP-Adapter (Apache-2.0) - InstantStyle weights + SDXL CLIP image encoder
* h94/IP-Adapter-FaceID - FaceID PlusV2 SDXL identity adapter weights
* InstantX/InstantID (Apache-2.0) - facial-keypoint ControlNet, converted to fp16
* public-data/insightface - buffalo_l ONNX pack for face embedding extraction
* LPDoctor/insightface mirror - antelopev2 ONNX pack for facial keypoints

All artifacts are written under ``models/instantstyle/`` with paths that mirror
the upstream repository layout so diffusers/insightface loaders can consume
them fully offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_MANIFEST = Path("models/instantstyle/manifest.json")
DEFAULT_MODEL_ROOT = Path("models/instantstyle")


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    prefix: str
    repo_id: str
    revision: str
    source_path: str
    convert: str = ""  # optional post-download conversion, e.g. "fp16"


SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
IP_ADAPTER_REVISION = "018e402774aeeddd60609b4ecdb7e298259dc729"
FACEID_REVISION = "43907e6f44d079bf1a9102d9a6e56aef7a219bae"
INSTANTID_REVISION = "57b32dfee076092ad2930c71fd6d439c2c3b1820"
INSIGHTFACE_REVISION = "33c1063c49c785b7652d3fd529f86fa4f149392b"
ANTELOPE_REVISION = "25226b4048397eb2adc0fa5a3c21f416005fc228"
CLIP_VIT_H_REVISION = "main"
CLIP_VIT_L_REVISION = "main"

_SDXL_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)

_IP_ADAPTER_FILES = (
    # The diffusers InstantStyle recipe pairs this adapter with the bigG
    # image encoder below; its projection consumes the encoder's pooled
    # 1280-d embeddings.
    "sdxl_models/ip-adapter_sdxl.safetensors",
    "sdxl_models/image_encoder/config.json",
    "sdxl_models/image_encoder/model.safetensors",
    "models/image_encoder/config.json",
    "models/image_encoder/model.safetensors",
)

_FACEID_FILES = ("ip-adapter-faceid-plusv2_sdxl.bin",)

# The InstantID ControlNet ships fp32 only. The engine container runs inside a
# WSL2 cgroup whose memory budget includes mapped model files, so the 2.5 GB
# fp32 checkpoint would cost a 2.5 GB fault-in burst at load time. Converting
# it to fp16 halves both the mapped footprint and the weight-activation
# transfer traffic; the conversion is recorded in the manifest.
_INSTANTID_FILES = (
    "ControlNetModel/config.json",
    "ControlNetModel/diffusion_pytorch_model.safetensors -> "
    "ControlNetModel/diffusion_pytorch_model.fp16.safetensors",
)

_ANTELOPE_FILES = (
    "models/antelopev2/1k3d68.onnx",
    "models/antelopev2/2d106det.onnx",
    "models/antelopev2/genderage.onnx",
    "models/antelopev2/glintr100.onnx",
    "models/antelopev2/scrfd_10g_bnkps.onnx",
)

_INSIGHTFACE_FILES = (
    "models/buffalo_l/1k3d68.onnx",
    "models/buffalo_l/2d106det.onnx",
    "models/buffalo_l/det_10g.onnx",
    "models/buffalo_l/genderage.onnx",
    "models/buffalo_l/w600k_r50.onnx",
)

# The h94 image-encoder folders ship weights only. Both encoders are
# standard OpenAI-CLIP vision towers (224px, OpenAI mean/std), so the
# CLIPImageProcessor config for each comes from openai/clip-vit-large-patch14;
# the original ViT-H host repo (laion/CLIP-ViT-H-14-laion2B-s39B-b160K) is no
# longer published on the Hub.
_CLIP_PREPROCESSOR_SOURCES = (
    ArtifactSource(
        "ip-adapter",
        "openai/clip-vit-large-patch14",
        CLIP_VIT_H_REVISION,
        "preprocessor_config.json -> sdxl_models/image_encoder/preprocessor_config.json",
    ),
    ArtifactSource(
        "ip-adapter",
        "openai/clip-vit-large-patch14",
        CLIP_VIT_L_REVISION,
        "preprocessor_config.json -> models/image_encoder/preprocessor_config.json",
    ),
)

SOURCES: tuple[ArtifactSource, ...] = (
    *(
        ArtifactSource(
            "sdxl-base", "stabilityai/stable-diffusion-xl-base-1.0", SDXL_REVISION, path
        )
        for path in _SDXL_FILES
    ),
    *(
        ArtifactSource("ip-adapter", "h94/IP-Adapter", IP_ADAPTER_REVISION, path)
        for path in _IP_ADAPTER_FILES
    ),
    *(
        ArtifactSource("faceid", "h94/IP-Adapter-FaceID", FACEID_REVISION, path)
        for path in _FACEID_FILES
    ),
    *(
        ArtifactSource(
            "instantid", "InstantX/InstantID", INSTANTID_REVISION, path,
            convert="fp16" if path.startswith("ControlNetModel/diffusion") else "",
        )
        for path in _INSTANTID_FILES
    ),
    *(
        ArtifactSource(
            "insightface", "public-data/insightface", INSIGHTFACE_REVISION, path
        )
        for path in _INSIGHTFACE_FILES
    ),
    *(
        ArtifactSource(
            "insightface", "LPDoctor/insightface", ANTELOPE_REVISION, path
        )
        for path in _ANTELOPE_FILES
    ),
    *_CLIP_PREPROCESSOR_SOURCES,
)

# The InstantID ControlNet checkpoint is converted to fp16 after download.
_CONTROLNET_FP16_TARGET = "instantid/ControlNetModel/diffusion_pytorch_model.fp16.safetensors"


def _needs_fp16_conversion(path: Path) -> bool:
    """Return True when a safetensors file is still fp32 and must be converted."""
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as opened:
        for key in opened.keys():
            return opened.get_slice(key).get_dtype() != "F16"
    return False


def _convert_to_fp16(path: Path) -> None:
    """Rewrite a safetensors checkpoint in place with fp16 weights."""
    import numpy as np
    from safetensors import numpy as np_safetensors

    tensors = {
        key: value.astype(np.float16)
        for key, value in np_safetensors.load_file(str(path)).items()
    }
    np_safetensors.save_file(tensors, str(path))


def _apply_conversions(model_root: Path) -> None:
    """Run post-download conversions (currently: ControlNet fp32 -> fp16)."""
    target = model_root / Path(*_CONTROLNET_FP16_TARGET.split("/"))
    if not target.is_file():
        return
    if _needs_fp16_conversion(target):
        print(f"converting {_CONTROLNET_FP16_TARGET} fp32 -> fp16")
        _convert_to_fp16(target)
    else:
        print(f"conversion already applied {_CONTROLNET_FP16_TARGET}")


def source_parts(source: ArtifactSource) -> tuple[str, str]:
    """Split a source path, supporting an optional ``source -> target`` remap."""
    parts = source.source_path.split(" -> ")
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0].strip(), parts[1].strip()


def local_path(model_root: Path, source: ArtifactSource) -> Path:
    _, target = source_parts(source)
    return model_root / source.prefix / Path(*PurePosixPath(target).parts)


def manifest_relative(source: ArtifactSource) -> str:
    _, target = source_parts(source)
    return (
        PurePosixPath(source.prefix)
        .joinpath(*PurePosixPath(target).parts)
        .as_posix()
    )


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_artifacts(model_root: Path, *, token: str | None, force: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Model download requires `huggingface-hub`; fp16 conversion requires "
            "`safetensors` and `numpy`. Install the provisioning dependencies "
            "first (uv run --with huggingface-hub --with safetensors --with numpy ...)."
        ) from exc

    for index, source in enumerate(SOURCES, 1):
        destination = local_path(model_root, source)
        if destination.is_file() and not force:
            print(f"[{index}/{len(SOURCES)}] present {manifest_relative(source)}")
            continue
        print(
            f"[{index}/{len(SOURCES)}] downloading {source.repo_id}:"
            f"{source.source_path} -> {destination}"
        )
        filename, _ = source_parts(source)
        downloaded = Path(
            hf_hub_download(
                repo_id=source.repo_id,
                filename=filename,
                revision=source.revision,
                local_dir=model_root / source.prefix,
                token=token,
                force_download=force,
            )
        )
        if downloaded.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(destination)
    _apply_conversions(model_root)


def verify_artifacts(model_root: Path) -> list[str]:
    failures: list[str] = []
    resolved_root = model_root.resolve()
    for source in SOURCES:
        path = local_path(model_root, source).resolve()
        relative = manifest_relative(source)
        if not path.is_relative_to(resolved_root):
            failures.append(f"unsafe resolved path: {relative}")
            continue
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing: {relative}")
            continue
        print(f"present {relative} bytes={path.stat().st_size}")
    return failures


def write_manifest(model_root: Path, manifest_path: Path) -> int:
    artifacts: list[dict[str, object]] = []
    for source in SOURCES:
        path = local_path(model_root, source)
        if not path.is_file():
            print(
                f"error: missing artifact before manifest write: {path}",
                file=sys.stderr,
            )
            return 1
        byte_size = path.stat().st_size
        sha256 = file_digest(path)
        source_filename, target_filename = source_parts(source)
        artifact: dict[str, object] = {
            "path": manifest_relative(source),
            "sha256": sha256,
            "bytes": byte_size,
            "source": f"https://huggingface.co/{source.repo_id}",
            "revision": source.revision,
            "source_path": source_filename,
            "target_path": target_filename,
        }
        if source.convert:
            artifact["conversion"] = source.convert
        artifacts.append(artifact)
        print(f"hashed {manifest_relative(source)} bytes={byte_size} sha256={sha256}")
    payload = {"schema_version": 1, "artifacts": artifacts}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} with {len(artifacts)} artifacts")
    return 0


def load_manifest_artifacts(manifest_path: Path) -> list[dict[str, object]]:
    payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("artifacts"), list)
        or not payload["artifacts"]
    ):
        raise ValueError(
            "manifest schema_version must be 1 with a non-empty artifacts list"
        )
    artifacts = payload["artifacts"]
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("each artifact must be an object")
    return artifacts  # type: ignore[return-value]


def verify_manifest(model_root: Path, manifest_path: Path) -> int:
    try:
        artifacts = load_manifest_artifacts(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    failures: list[str] = []
    resolved_root = model_root.resolve()
    for artifact in artifacts:
        relative = str(artifact.get("path", ""))
        path = (model_root / Path(*PurePosixPath(relative).parts)).resolve()
        if not path.is_relative_to(resolved_root):
            failures.append(f"unsafe resolved path: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        expected_bytes = int(artifact.get("bytes", -1))  # type: ignore[arg-type]
        expected_sha256 = str(artifact.get("sha256", "")).lower()
        if path.stat().st_size != expected_bytes:
            failures.append(
                f"size mismatch: {relative} expected={expected_bytes} "
                f"actual={path.stat().st_size}"
            )
            continue
        actual = file_digest(path)
        if actual != expected_sha256:
            failures.append(
                f"checksum mismatch: {relative} expected={expected_sha256} actual={actual}"
            )
            continue
        print(f"verified {relative} bytes={expected_bytes}")
    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 1
    print(f"verified {len(artifacts)} InstantStyle runtime artifacts")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download", action="store_true", help="download missing artifacts"
    )
    mode.add_argument(
        "--write-manifest",
        action="store_true",
        help="hash local artifacts and rewrite the manifest",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verify local artifacts against the manifest without network access",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download existing artifacts"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="optional Hugging Face token; prefer the HF_TOKEN environment variable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.force and not args.download:
        print("error: --force requires --download", file=sys.stderr)
        return 2
    if args.download:
        import os

        token = args.hf_token or os.environ.get("HF_TOKEN")
        try:
            download_artifacts(args.model_root, token=token, force=args.force)
            failures = verify_artifacts(args.model_root)
        except (OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if failures:
            for failure in failures:
                print(f"error: {failure}", file=sys.stderr)
            return 1
        print(
            "all InstantStyle artifacts are present; run --write-manifest to refresh hashes"
        )
        return 0
    if args.write_manifest:
        return write_manifest(args.model_root, args.manifest)
    return verify_manifest(args.model_root, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
