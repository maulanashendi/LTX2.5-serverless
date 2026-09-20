from __future__ import annotations

import asyncio
import time
from typing import Any


async def check_server(
    session: Any,
    host: str,
    timeout_s: float = 600.0,
    interval_s: float = 1.0,
) -> None:
    """Poll GET /system_stats until ComfyUI answers 2xx.

    Uses an attempt count derived from timeout_s/interval_s rather than a wall-clock
    deadline so tests can patch asyncio.sleep without also having to patch time.
    """
    attempts = max(1, int(timeout_s / interval_s))
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            async with session.get(f"http://{host}/system_stats") as resp:
                if resp.status < 300:
                    return
                last_error = RuntimeError(f"status {resp.status}")
        except Exception as exc:  # noqa: BLE001 - any transport failure is a retry signal
            last_error = exc

        if attempt < attempts - 1:
            await asyncio.sleep(interval_s)

    detail = f" (last error: {last_error})" if last_error else ""
    raise TimeoutError(
        f"LTX_COMFY_UNREACHABLE: ComfyUI at {host} did not become ready "
        f"within {timeout_s:g}s{detail}"
    )


async def queue_prompt(session: Any, host: str, workflow: dict[str, Any]) -> str:
    async with session.post(f"http://{host}/prompt", json={"prompt": workflow}) as resp:
        body = await resp.json()

    prompt_id = body.get("prompt_id") if isinstance(body, dict) else None
    if prompt_id:
        return prompt_id

    node_errors = body.get("node_errors") if isinstance(body, dict) else None
    if node_errors:
        raise RuntimeError(
            "LTX_GRAPH_REJECTED: ComfyUI rejected the graph "
            f"(offending nodes: {', '.join(sorted(node_errors))}): {body}"
        )
    raise RuntimeError(f"LTX_GRAPH_REJECTED: ComfyUI returned no prompt_id: {body}")


async def poll_history(
    session: Any,
    host: str,
    prompt_id: str,
    start_time: float,
    timeout_s: float = 900.0,
    interval_s: float = 2.0,
) -> dict[str, Any]:
    fail_count = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_s:
            raise TimeoutError(
                f"LTX_RENDER_TIMEOUT: render exceeded {timeout_s:g}s for prompt {prompt_id}"
            )

        try:
            async with session.get(f"http://{host}/history/{prompt_id}") as resp:
                history = await resp.json()
        except Exception:  # noqa: BLE001 - transient disconnect, retry until fail_count trips
            fail_count += 1
            if fail_count > 5:
                raise RuntimeError(
                    f"LTX_COMFY_UNREACHABLE: lost connection to ComfyUI at {host} "
                    "while polling history"
                )
            await asyncio.sleep(interval_s)
            continue

        if prompt_id in history:
            return history[prompt_id]

        await asyncio.sleep(min(interval_s + elapsed / 30, 5))


async def run_graph(
    session: Any,
    host: str,
    workflow: dict[str, Any],
    start_time: float,
    ready_timeout_s: float = 600.0,
    render_timeout_s: float = 900.0,
) -> dict[str, Any]:
    await check_server(session, host, timeout_s=ready_timeout_s)
    prompt_id = await queue_prompt(session, host, workflow)
    return await poll_history(session, host, prompt_id, start_time, timeout_s=render_timeout_s)
