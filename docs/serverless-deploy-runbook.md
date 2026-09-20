# Deploying the worker to Runpod Serverless

Order matters: the endpoint can only reference an image that already exists, and a
network volume cannot be moved between data centers once created.

## 0. Prerequisites

- A Runpod API key in `~/.runpod/config.toml` or `RUNPOD_API_KEY`.
- A Docker Hub account with a Read & Write access token.
- A Hugging Face token whose account has accepted the gated
  [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) licence.

## 1. Publish an image

The build runs in GitHub Actions and pushes to Docker Hub. It needs four repository
settings that are **not currently configured** — without them the build step cannot
log in, and no image is produced:

```bash
gh variable set DOCKERHUB_REPO --body "<dockerhub-namespace>"
gh variable set DOCKERHUB_IMG  --body "ltx-2.5-serverless"
gh secret   set DOCKERHUB_USERNAME --body "<dockerhub-username>"
gh secret   set DOCKERHUB_TOKEN    --body "<dockerhub-access-token>"
```

Then build a tag:

```bash
gh workflow run manual-push-dockerhub.yml -f target_to_build=ltx2-5-distilled-int8
gh run watch
```

`ltx2-5-distilled-int8` targets CUDA 13.0. Use `ltx2-5-distilled-int8-cu128` for a
CUDA 12.8 host. The image carries code only — roughly 4 GB; model weights are not
baked in (see step 2).

## 2. Create a network volume

The distilled-int8 profile pulls about 50 GB of weights:

| File | Size |
| --- | --- |
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21.5 GB |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.4 GB |
| `gemma4_e2b_it_bf16.safetensors` (prompt enhancer) | 10.3 GB |
| video VAE, audio VAE, latent upscaler | 2.9 GB |

Baking these into the image is not practical: the result is ~54 GB, which exceeds the
disk on a GitHub-hosted runner. Keep them on a network volume instead, so the download
happens once rather than on every cold start.

```bash
curl -sS -X POST https://rest.runpod.io/v1/networkvolumes \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"ltx25-models","size":150,"dataCenterId":"<DC>"}'
```

Size 150 GB leaves room for the weights plus the seeded ComfyUI tree, the virtualenv,
and the HF/pip/torch caches that `bootstrap_workspace.sh` puts under
`<volume>/worker-comfyui/cache`. **The volume and the endpoint must be in the same data
center**, and the volume cannot be moved later.

## 3. Create the template and endpoint

Use `PERSIST_WORKSPACE=true` so the bootstrap symlinks `/comfyui`, `/opt/venv` and
`/comfyui/models` onto the volume and regenerates `extra_model_paths.yaml` against it.

```bash
IMAGE=<namespace>/ltx-2.5-serverless:<tag> \
NETWORK_VOLUME_ID=<volume-id> \
DATACENTER=<DC> \
HUGGINGFACE_ACCESS_TOKEN=<hf-token> \
  python3 scripts/create_serverless_endpoint.py
```

Settings that matter:

| Setting | Value | Why |
| --- | --- | --- |
| `executionTimeoutMs` | `1200000` | The handler's own render timeout is 900 s and its HTTP timeout 1000 s. The existing endpoints use 600000, so a slow render is killed before the handler can report. |
| `COMFY_READY_TIMEOUT` | `900` | First boot downloads the weights before ComfyUI answers. |
| `LTX_FRONTEND_ENABLED` | `false` | The bundled frontend is for pod mode; a serverless worker does not serve it. |
| `PERSIST_WORKSPACE` | `true` | Required for the volume to be used at all. |
| `workersMin` | `0` | Scale to zero between jobs. |

## 4. Verify, cheapest check first

```bash
EP=<endpoint-id>
# 1. Health check — returns before touching ComfyUI, so it proves only that the
#    worker booted and the handler is registered.
curl -sS -X POST "https://api.runpod.ai/v2/$EP/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"health_check":true}}'

# 2. Text-to-video. The first call also downloads ~50 GB onto the volume, so submit it
#    asynchronously and poll rather than blocking on /runsync.
JOB=$(curl -sS -X POST "https://api.runpod.ai/v2/$EP/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"prompt":"A lone warrior walking across a vast desert at dusk.","duration":5,"aspect_ratio":"16:9"}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
watch -n 15 "curl -sS -H 'Authorization: Bearer $RUNPOD_API_KEY' https://api.runpod.ai/v2/$EP/status/$JOB"

# 3. Image-to-video — same call with an "image" data URL added.
```

A `COMPLETED` status is not success on its own: check that `output.status == "success"`.
A failure now surfaces as RunPod status `FAILED` with the message in `error`, carrying
one of the `LTX_COMFY_UNREACHABLE`, `LTX_GRAPH_REJECTED`, `LTX_RENDER_TIMEOUT` or
`LTX_NO_OUTPUT` prefixes.

## 5. Point the senai backend at it

The endpoint ID is registered through the senai admin panel, not an environment
variable — it resolves the alias `ltx25-main/i2v` to an endpoint binding. The backend
rejects inline base64 output, so set `AWS_BUCKET_NAME` and its credentials on the
template if the backend is to consume this endpoint.

## GPU note

`.runpod/hub.json` requests `BLACKWELL_180`, and the `int8-convrot` weights are
published as Blackwell-oriented. The endpoints configured today use A40 and RTX A6000,
which are Ampere. If the int8 kernels refuse to load there, it will show up as a
ComfyUI node error on the first real render — a model-format problem that no endpoint
setting can work around.

## Cost and cleanup

- A network volume bills monthly whether or not a worker is running.
- `workersMin: 0` means no GPU cost while idle; the first request after idle pays a
  cold start.
- Delete an endpoint and its template when finished; delete the volume separately —
  removing the endpoint does not remove it.
