# ==============================================================================
# LTX 2.5 RunPod serverless handler
#
# Dispatches jobs to one of two paths:
#   - "workflow": a raw ComfyUI API-format graph supplied by the caller
#   - flat: {"prompt", "image"?, ...} — the graph is built from ltx_graph
# Exceptions propagate out of handler() so the RunPod SDK marks the job FAILED
# instead of masking failures behind {"status": "error"}.
# ==============================================================================

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
import boto3
import runpod
from botocore.config import Config
import redis.asyncio as redis

import comfy_client
from ltx_graph import DEFAULT_IMAGE_NAME, build_graph
from workflow_support import (
    apply_input_filename_map,
    build_output_path,
    build_workflow_cache_key,
    collect_output_entries,
    guess_media_type,
    is_workflow_job,
    materialize_image,
    write_input_images,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s"}',
)
logger = logging.getLogger("ltx25-worker")

REDIS_URL = "redis://127.0.0.1:6379"
if os.environ.get("REDIS_URL", REDIS_URL) not in {
    f"redis://{host}:6379{suffix}"
    for host in ("127.0.0.1", "localhost")
    for suffix in ("", "/", "/0")
}:
    raise RuntimeError("External Redis is disabled; unset REDIS_URL to use local Redis.")
redis_client = redis.from_url(
    REDIS_URL, decode_responses=True, socket_connect_timeout=5, retry_on_timeout=True
)

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_READY_TIMEOUT = int(os.environ.get("COMFY_READY_TIMEOUT", "600"))
COMFY_INPUT_DIR = os.environ.get("COMFY_INPUT_DIR", "/comfyui/input")
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output")
MAX_INLINE_VIDEO_MB = int(os.environ.get("MAX_INLINE_VIDEO_MB", "50"))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "604800"))

_comfy_ready = False


async def redis_safe(coro: Any, default: Any = None) -> Any:
    """Await a Redis coroutine, degrading to `default` on any Redis failure."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 - Redis is best-effort everywhere it's used
        logger.warning(f"Redis operation failed, continuing without it: {exc}")
        return default


def decode_cached_response(raw_value: str | None) -> dict | None:
    if raw_value is None:
        return None
    try:
        cached_response = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(cached_response, dict) or cached_response.get("status") != "success":
        return None

    response = dict(cached_response)
    response["cached"] = True
    return response


async def _release_lock(cache_hash: str, lock_token: str) -> None:
    current = await redis_client.get(cache_hash)
    if current == lock_token:
        await redis_client.delete(cache_hash)


def success_envelope(output: dict, start_time: float, **extra: Any) -> dict:
    return {
        "status": "success",
        "output": output,
        "metadata": {
            "render_time_sec": round(time.time() - start_time, 2),
            "node_used": COMFY_HOST,
            **extra,
        },
    }


def build_job_image_inputs(
    job_id: str, images: list[dict[str, str]] | None
) -> tuple[dict[str, str], list[dict[str, str]]]:
    if not images:
        return {}, []

    replacements: dict[str, str] = {}
    prepared_images: list[dict[str, str]] = []

    for image in images:
        original_name = image.get("name")
        image_data = image.get("image")
        if not original_name or not image_data:
            raise ValueError(
                "'images' must be a list of objects with 'name' and 'image' keys."
            )

        unique_name = str(Path(job_id) / Path(original_name)).replace("\\", "/")
        replacements[original_name] = unique_name
        prepared_images.append({"name": unique_name, "image": image_data})

    return replacements, prepared_images


def cleanup_input_files(filepaths: list[str]) -> None:
    input_root = Path(COMFY_INPUT_DIR).resolve()
    for filepath in filepaths:
        try:
            path = Path(os.path.realpath(filepath))
            if not str(path).startswith(str(input_root) + os.sep):
                continue
            if path.exists():
                path.unlink()
        except OSError:
            logger.warning(f"Failed to clean up input file: {filepath}")

    for filepath in filepaths:
        parent = Path(os.path.realpath(filepath)).parent
        while str(parent).startswith(str(input_root) + os.sep) and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


async def upload_to_s3_with_retry(
    filepath: str, storage_key: str, content_type: str | None = None
) -> str:
    bucket = os.environ.get("AWS_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("AWS_BUCKET_NAME missing.")

    boto_config = Config(retries={"max_attempts": 3, "mode": "standard"})

    def _upload():
        s3 = boto3.client("s3", config=boto_config)
        extra_args = {"ContentType": content_type} if content_type else None
        if extra_args:
            s3.upload_file(filepath, bucket, storage_key, ExtraArgs=extra_args)
        else:
            s3.upload_file(filepath, bucket, storage_key)
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": storage_key},
            ExpiresIn=604800,
        )

    for attempt in range(3):
        try:
            return await asyncio.to_thread(_upload)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2**attempt)


async def build_output_entry(
    filepath: str,
    job_id: str,
    entry: dict[str, str],
    index: int,
) -> dict:
    media_type = guess_media_type(entry["filename"], entry["media_kind"])
    path = Path(filepath)

    if entry["media_kind"] == "video":
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_INLINE_VIDEO_MB and not os.environ.get("AWS_BUCKET_NAME"):
            raise RuntimeError(
                f"Video output is {file_size_mb:.1f}MB, which exceeds "
                f"MAX_INLINE_VIDEO_MB={MAX_INLINE_VIDEO_MB}. "
                "Configure S3 upload or raise the inline limit."
            )

    if os.environ.get("AWS_BUCKET_NAME"):
        storage_key = f"renders/{job_id}/{index:02d}-{path.name}"
        data = await upload_to_s3_with_retry(filepath, storage_key, media_type)
        output_type = "url"
    else:
        with open(filepath, "rb") as file_handle:
            data = base64.b64encode(file_handle.read()).decode("utf-8")
        output_type = "base64"

    return {
        "filename": entry["filename"],
        "subfolder": entry.get("subfolder", ""),
        "type": output_type,
        "data": data,
        "media_type": media_type,
    }


async def build_workflow_output_payload(history_entry: dict, job_id: str) -> dict:
    entries = collect_output_entries(history_entry.get("outputs", {}))
    if not entries:
        raise RuntimeError("LTX_NO_OUTPUT: workflow finished without image or video output.")

    output: dict[str, list[dict]] = {"images": [], "videos": []}
    for index, entry in enumerate(entries):
        output_path = build_output_path(COMFY_OUTPUT_DIR, entry)
        if not output_path.exists():
            raise RuntimeError(f"Expected output file missing: {output_path}")

        payload = await build_output_entry(str(output_path), job_id, entry, index)
        collection = "images" if entry["media_kind"] == "image" else "videos"
        output[collection].append(payload)

    return {key: value for key, value in output.items() if value}


async def wait_for_comfy_ready(
    session: aiohttp.ClientSession,
    timeout_s: float = COMFY_READY_TIMEOUT,
    interval_s: float = 1.0,
) -> None:
    global _comfy_ready
    if _comfy_ready:
        return
    await comfy_client.check_server(session, COMFY_HOST, timeout_s=timeout_s, interval_s=interval_s)
    _comfy_ready = True


async def execute_workflow(
    session: aiohttp.ClientSession, workflow: dict, job_id: str, start_time: float
) -> dict:
    await wait_for_comfy_ready(session)
    prompt_id = await comfy_client.queue_prompt(session, COMFY_HOST, workflow)
    return await comfy_client.poll_history(session, COMFY_HOST, prompt_id, start_time)


async def handle_workflow_job(job_id: str, job_input: dict, start_time: float) -> dict:
    workflow = job_input.get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError("Missing 'workflow'.")

    images = job_input.get("images")
    cache_hash = build_workflow_cache_key(workflow, images)
    lock_token = str(uuid.uuid4())

    redis_state = await redis_safe(redis_client.get(cache_hash))
    cached_response = decode_cached_response(redis_state)
    if cached_response:
        await redis_safe(
            redis_client.hset(
                f"job_status:{job_id}",
                mapping={"status": "completed", "cache_hit": "true"},
            )
        )
        return cached_response

    lock_acquired = await redis_safe(redis_client.set(cache_hash, lock_token, ex=1200, nx=True))
    if lock_acquired is False:
        logger.info(f"[{job_id}] Deduplication active; waiting up to 60s.")
        await redis_safe(
            redis_client.hset(f"job_status:{job_id}", mapping={"status": "waiting_in_queue"})
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            await asyncio.sleep(5)
            new_state = await redis_safe(redis_client.get(cache_hash))
            cached_response = decode_cached_response(new_state)
            if cached_response:
                return cached_response
        logger.info(f"[{job_id}] Deduplication wait expired; rendering anyway.")

    written_input_files: list[str] = []
    try:
        name_map, prepared_images = build_job_image_inputs(job_id, images)
        prepared_workflow = apply_input_filename_map(workflow, name_map)
        written_input_files = write_input_images(COMFY_INPUT_DIR, prepared_images)

        http_timeout = aiohttp.ClientTimeout(total=1000)
        async with aiohttp.ClientSession(timeout=http_timeout) as session:
            history_entry = await execute_workflow(session, prepared_workflow, job_id, start_time)

        await redis_safe(redis_client.hset(f"job_status:{job_id}", mapping={"status": "uploading"}))
        output_payload = await build_workflow_output_payload(history_entry, job_id)
        response = success_envelope(output_payload, start_time)

        current_lock = await redis_safe(redis_client.get(cache_hash))
        if current_lock == lock_token:
            await redis_safe(
                redis_client.set(cache_hash, json.dumps(response), ex=CACHE_TTL_SECONDS)
            )

        return response
    finally:
        cleanup_input_files(written_input_files)
        await redis_safe(_release_lock(cache_hash, lock_token))


async def handle_flat_job(job_id: str, job_input: dict, start_time: float) -> dict:
    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt is required.")

    image = job_input.get("image")
    inferred_mode = "i2v" if image else "t2v"
    requested_mode = job_input.get("mode")
    if requested_mode is not None and requested_mode != inferred_mode:
        raise ValueError(
            f"mode={requested_mode!r} conflicts with image presence "
            f"(image {'present' if image else 'absent'} implies mode={inferred_mode!r})."
        )
    mode = inferred_mode

    image_name = job_input.get("image_name") or DEFAULT_IMAGE_NAME
    duration = job_input.get("duration", 5)
    aspect_ratio = job_input.get("aspect_ratio", "16:9")
    optimize_prompt = job_input.get("optimize_prompt", True)
    seed = job_input.get("seed")

    written_input_files: list[str] = []
    http_timeout = aiohttp.ClientTimeout(total=1000)
    try:
        # The session is opened before the image is materialized so that a remote
        # image URL has a client to download with.
        async with aiohttp.ClientSession(timeout=http_timeout) as session:
            image_ref = None
            if mode == "i2v":
                image_ref, written_path = await materialize_image(
                    COMFY_INPUT_DIR, job_id, image, image_name, session=session
                )
                written_input_files.append(written_path)

            graph = build_graph(
                mode=mode,
                prompt=prompt,
                seconds=duration,
                aspect_ratio=aspect_ratio,
                image_name=image_ref,
                optimize_prompt=optimize_prompt,
                seed=seed,
            )

            history_entry = await execute_workflow(session, graph, job_id, start_time)

        output_payload = await build_workflow_output_payload(history_entry, job_id)
        return success_envelope(output_payload, start_time)
    finally:
        cleanup_input_files(written_input_files)


async def handler(job: dict) -> dict:
    job_input = job.get("input", {}) or {}

    if job_input.get("health_check") is True:
        return {"status": "healthy", "service": "ltx-2.5-worker"}

    job_id = job.get("id", uuid.uuid4().hex)
    start_time = time.time()

    await redis_safe(
        redis_client.hset(
            f"job_status:{job_id}", mapping={"status": "initializing", "progress": "0%"}
        )
    )
    await redis_safe(redis_client.expire(f"job_status:{job_id}", 3600))

    try:
        if is_workflow_job(job_input):
            response = await handle_workflow_job(job_id, job_input, start_time)
        else:
            response = await handle_flat_job(job_id, job_input, start_time)
    except Exception as exc:
        await redis_safe(
            redis_client.hset(f"job_status:{job_id}", mapping={"status": "failed", "error": str(exc)})
        )
        raise

    await redis_safe(redis_client.hset(f"job_status:{job_id}", mapping={"status": "completed"}))
    return response


logger.info("Initializing LTX 2.5 serverless worker...")
runpod.serverless.start({"handler": handler})
