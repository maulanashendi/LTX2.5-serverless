#!/usr/bin/env python3
"""Create a template and serverless endpoint for the LTX 2.5 worker.

Creates billable resources. See docs/serverless-deploy-runbook.md.

Required env: IMAGE, NETWORK_VOLUME_ID, DATACENTER, HUGGINGFACE_ACCESS_TOKEN
Optional env: NAME, GPU_TYPE_IDS (JSON list), CONTAINER_DISK_GB, WORKERS_MAX,
              EXECUTION_TIMEOUT_MS, RUNPOD_API_KEY
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    config = Path.home() / ".runpod" / "config.toml"
    if config.exists():
        match = re.search(r'^\s*api_key\s*=\s*"([^"]+)"', config.read_text(), re.M)
        if match:
            return match.group(1)
    sys.exit("No Runpod API key: set RUNPOD_API_KEY or run 'flash login'.")


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Set {name}. See docs/serverless-deploy-runbook.md.")
    return value


def post(path: str, payload: dict, key: str) -> dict:
    request = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        sys.exit(f"POST {path} failed: HTTP {exc.code}\n{exc.read().decode()[:800]}")


def main() -> None:
    key = api_key()
    image = required("IMAGE")
    volume_id = required("NETWORK_VOLUME_ID")
    datacenter = required("DATACENTER")
    hf_token = required("HUGGINGFACE_ACCESS_TOKEN")

    name = os.environ.get("NAME", "ltx25-worker")
    gpu_type_ids = json.loads(
        os.environ.get("GPU_TYPE_IDS", '["NVIDIA A40", "NVIDIA RTX A6000"]')
    )

    print(f"==> template from {image}")
    template = post(
        "/templates",
        {
            "name": f"{name}-tpl",
            "imageName": image,
            "isServerless": True,
            # Code image only — the weights live on the network volume.
            "containerDiskInGb": int(os.environ.get("CONTAINER_DISK_GB", "30")),
            "env": {
                "RUN_MODE": "worker",
                "PERSIST_WORKSPACE": "true",
                "LTX25_PRELOAD_VARIANT": "distilled-int8",
                "LTX25_PRELOAD_PROMPT_ENHANCER": "true",
                "LTX_FRONTEND_ENABLED": "false",
                "COMFY_HOST": "127.0.0.1:8188",
                # First boot pulls ~50GB before ComfyUI answers.
                "COMFY_READY_TIMEOUT": "900",
                "HUGGINGFACE_ACCESS_TOKEN": hf_token,
            },
        },
        key,
    )
    template_id = template["id"]
    print(f"    templateId={template_id}")

    print("==> endpoint")
    endpoint = post(
        "/endpoints",
        {
            "name": name,
            "templateId": template_id,
            "gpuTypeIds": gpu_type_ids,
            "gpuCount": 1,
            "dataCenterIds": [datacenter],
            "networkVolumeId": volume_id,
            "workersMin": 0,
            "workersMax": int(os.environ.get("WORKERS_MAX", "1")),
            "idleTimeout": 5,
            "flashboot": True,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
            # The handler's own render timeout is 900s and its HTTP timeout 1000s,
            # so anything below that kills a slow render before it can report.
            "executionTimeoutMs": int(os.environ.get("EXECUTION_TIMEOUT_MS", "1200000")),
        },
        key,
    )
    print(json.dumps(endpoint, indent=2))
    endpoint_id = endpoint["id"]

    print(
        f"\nendpointId={endpoint_id}\n\n"
        "Smoke test (health check only, no GPU work):\n"
        f"  curl -sS -X POST https://api.runpod.ai/v2/{endpoint_id}/runsync \\\n"
        '    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \\\n'
        "    -d '{\"input\":{\"health_check\":true}}'\n\n"
        "The first render downloads ~50GB onto the volume. Submit it with /run and\n"
        "poll /status rather than blocking on /runsync — runbook step 4."
    )


if __name__ == "__main__":
    main()
