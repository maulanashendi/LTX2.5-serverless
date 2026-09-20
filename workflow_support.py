from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import ipaddress
import json
import mimetypes
import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import ltx_graph

OUTPUT_KEYS = {
    "images": "image",
    "videos": "video",
    "gifs": "video",
}

ALLOW_REMOTE_IMAGE_ENV = "LTX_ALLOW_REMOTE_IMAGE"
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def is_workflow_job(job_input: dict[str, Any]) -> bool:
    return isinstance(job_input.get("workflow"), dict)


def build_workflow_cache_key(
    workflow: dict[str, Any], images: list[dict[str, str]] | None
) -> str:
    normalized = {"workflow": workflow, "images": images or []}
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decode_base64_data(data: str) -> bytes:
    if "," in data and data.split(",", 1)[0].endswith(";base64"):
        data = data.split(",", 1)[1]

    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def safe_input_path(base_dir: str, image_name: str) -> Path:
    relative_path = Path(image_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Input image name must be a safe relative path.")

    target = (Path(base_dir) / relative_path).resolve()
    root = Path(base_dir).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Input image path escapes COMFY_INPUT_DIR.")

    return target


def write_input_images(
    base_dir: str, images: list[dict[str, str]] | None
) -> list[str]:
    if not images:
        return []

    written_files: list[str] = []
    for image in images:
        name = image.get("name")
        data = image.get("image")
        if not name or not data:
            raise ValueError(
                "'images' must be a list of objects with 'name' and 'image' keys."
            )

        target = safe_input_path(base_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(decode_base64_data(data))
        written_files.append(str(target))

    return written_files


def apply_input_filename_map(
    workflow: dict[str, Any], replacements: dict[str, str]
) -> dict[str, Any]:
    def _replace(value: Any) -> Any:
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [_replace(item) for item in value]
        if isinstance(value, dict):
            return {key: _replace(item) for key, item in value.items()}
        return value

    return _replace(copy.deepcopy(workflow))


def collect_output_entries(outputs: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue

        for output_key, media_kind in OUTPUT_KEYS.items():
            files = node_output.get(output_key, [])
            if not isinstance(files, list):
                continue

            for file_info in files:
                filename = file_info.get("filename")
                if not filename:
                    continue

                entries.append(
                    {
                        "filename": filename,
                        "subfolder": file_info.get("subfolder", ""),
                        "media_kind": media_kind,
                    }
                )

    return entries


def build_output_path(base_dir: str, entry: dict[str, str]) -> Path:
    subfolder = entry.get("subfolder") or ""
    target = (Path(base_dir) / subfolder / entry["filename"]).resolve()
    root = Path(base_dir).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Output path escapes COMFY_OUTPUT_DIR.")
    return target


def guess_media_type(filename: str, media_kind: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed
    return "image/png" if media_kind == "image" else "video/mp4"


def _reject_unsafe_host(hostname: str) -> None:
    try:
        resolved = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve image host: {hostname}") from exc

    for _family, _type, _proto, _canonname, sockaddr in resolved:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Refusing to fetch image from unsafe host: {hostname}")


async def _fetch_remote_image(url: str, session: Any) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http:// and https:// image URLs are supported.")
    if not parsed.hostname:
        raise ValueError("Image URL is missing a hostname.")

    _reject_unsafe_host(parsed.hostname)

    if session is None:
        raise ValueError("An HTTP session is required to download a remote image.")

    async with session.get(url, allow_redirects=False) as response:
        if not (200 <= response.status < 300):
            raise ValueError(f"Failed to download remote image: HTTP {response.status}")

        chunks: list[bytes] = []
        total = 0
        async for chunk in response.content.iter_chunked(65536):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("Remote image exceeds the maximum allowed size.")
            chunks.append(chunk)

        return b"".join(chunks)


async def materialize_image(
    base_dir: str,
    job_id: str,
    image: str,
    image_name: str,
    session: Any = None,
) -> tuple[str, str]:
    safe = ltx_graph.sanitize_image_name(image_name or ltx_graph.DEFAULT_IMAGE_NAME)
    scoped = f"{job_id}/{safe}"
    target = safe_input_path(base_dir, scoped)

    if image.startswith("http://") or image.startswith("https://"):
        if os.environ.get(ALLOW_REMOTE_IMAGE_ENV) != "true":
            raise ValueError(
                f"Remote image URLs are disabled; set {ALLOW_REMOTE_IMAGE_ENV}=true to allow."
            )
        data = await _fetch_remote_image(image, session)
    else:
        data = decode_base64_data(image)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return scoped, str(target)
