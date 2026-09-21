#!/usr/bin/env python3
"""Create a template and serverless endpoint for the LTX 2.5 worker.

Creates billable resources. See docs/serverless-deploy-runbook.md.

Required env: IMAGE, LTX25_REPO_URL
Optional env: NAME, GPU_TYPE_IDS (JSON list), DATACENTERS (JSON list),
              CONTAINER_DISK_GB, WORKERS_MAX, EXECUTION_TIMEOUT_MS, RUNPOD_API_KEY

Model weights are pulled from LTX25_REPO_URL (an R2 mirror) onto container disk
at cold start. There is no network volume, so the endpoint is not pinned to one
data center.
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
    repo_url = required("LTX25_REPO_URL")

    name = os.environ.get("NAME", "ltx25-worker")
    datacenters = json.loads(os.environ.get("DATACENTERS", "[]"))
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
            # Holds the image plus the ~50GB of weights pulled at cold start.
            "containerDiskInGb": int(os.environ.get("CONTAINER_DISK_GB", "120")),
            "env": {
                "RUN_MODE": "worker",
                # Keep this "true" even with no volume attached. bootstrap_workspace
                # then takes its "no persistent mount detected" branch, which still
                # regenerates extra_model_paths.yaml against /comfyui. Setting it
                # "false" returns before that write and leaves the baked-in file
                # pointing at a /runpod-volume that does not exist.
                "PERSIST_WORKSPACE": "true",
                "LTX25_PRELOAD_VARIANT": "distilled-int8",
                "LTX25_PRELOAD_PROMPT_ENHANCER": "true",
                # Weights come from the R2 mirror, not HuggingFace, so a cold start
                # does not depend on HF being up or under its rate limit.
                "LTX25_REPO_URL": repo_url,
                "LTX_FRONTEND_ENABLED": "false",
                "COMFY_HOST": "127.0.0.1:8188",
                # First boot pulls ~50GB before ComfyUI answers.
                "COMFY_READY_TIMEOUT": "900",
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
            # No network volume, so the endpoint may span data centers. An empty
            # list lets Runpod place workers wherever the GPUs are available.
            **({"dataCenterIds": datacenters} if datacenters else {}),
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
        "Every cold start pulls ~50GB from the mirror onto container disk, so\n"
        "submit the first render with /run and poll /status rather than blocking\n"
        "on /runsync — runbook step 4."
    )


if __name__ == "__main__":
    main()
