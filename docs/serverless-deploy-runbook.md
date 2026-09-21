# Deploying the worker to Runpod Serverless

Order matters: the endpoint can only reference an image that already exists, and a
network volume cannot be moved between data centers once created.

## 0. Prerequisites

- A Runpod API key in `~/.runpod/config.toml` or `RUNPOD_API_KEY`.
- A Docker Hub account with a Read & Write access token.
- A Hugging Face token whose account has accepted the gated
  [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) licence. This is
  needed **once**, to mirror the weights (step 2) — the running worker never talks to
  HuggingFace.
- A Cloudflare R2 bucket and an S3 API key pair for it.

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

## 2. Mirror the weights to R2

The distilled-int8 profile needs about 50 GB:

| File | Size |
| --- | --- |
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21.5 GB |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.4 GB |
| `gemma4_e2b_it_bf16.safetensors` (prompt enhancer) | 10.3 GB |
| video VAE, audio VAE, latent upscaler | 2.9 GB |

Three options were weighed, and the weights are served from **Cloudflare R2**:

- **Baked into the image** — the result is ~54 GB, more disk than a GitHub-hosted
  runner has. Not possible with the current CI.
- **Runpod network volume** — fast cold starts, but a volume pins the endpoint to a
  single data center. With A40 / RTX A6000 that is a real availability risk.
- **R2 mirror (chosen)** — the worker pulls the weights at cold start. No data center
  lock, no dependency on HuggingFace being up or under its rate limit at boot, and R2
  egress is free.

The trade: every cold start pays GPU time while it downloads. Raise `workersMin` to 1
if that cost outweighs the idle cost for your traffic.

Mirror once, from your own machine:

```bash
R2_BUCKET=<bucket> \
R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com \
AWS_ACCESS_KEY_ID=<r2-access-key> \
AWS_SECRET_ACCESS_KEY=<r2-secret> \
HUGGINGFACE_ACCESS_TOKEN=<hf-token> \
  ./scripts/mirror-models-to-r2.sh
```

The script prints the `LTX25_REPO_URL` to set on the template. The worker expects the
same relative layout as the HuggingFace repo — `diffusion_models/`, `text_encoders/`,
`vae/`, `latent_upscale_models/` — so the mirror is a straight copy, not a re-layout.

> `Lightricks/LTX-2.5` is a gated model. Confirm its licence permits mirroring to your
> own bucket, private or not, before running this.

## 3. Create the template and endpoint

```bash
IMAGE=<namespace>/ltx-2.5-serverless:<tag> \
LTX25_REPO_URL=https://<your-r2-public-host>/ltx-2.5 \
  python3 scripts/create_serverless_endpoint.py
```

Leave `DATACENTERS` unset so Runpod places workers wherever the GPUs are free — with
no volume attached there is nothing tying the endpoint to one region.

Settings that matter:

| Setting | Value | Why |
| --- | --- | --- |
| `executionTimeoutMs` | `1200000` | The handler's own render timeout is 900 s and its HTTP timeout 1000 s. The existing endpoints use 600000, so a slow render is killed before the handler can report. |
| `COMFY_READY_TIMEOUT` | `900` | First boot downloads the weights before ComfyUI answers. |
| `LTX_FRONTEND_ENABLED` | `false` | The bundled frontend is for pod mode; a serverless worker does not serve it. |
| `PERSIST_WORKSPACE` | `true` | Counter-intuitive with no volume, but correct. `bootstrap_workspace` then takes its "no persistent mount detected" branch, which still regenerates `extra_model_paths.yaml` against `/comfyui`. Setting it `false` returns *before* that write and leaves the baked-in file pointing at a `/runpod-volume` that does not exist. |
| `LTX25_REPO_URL` | your R2 mirror | Where the weights come from. Unset means HuggingFace, which is what this setup exists to avoid at boot. |
| `containerDiskInGb` | `120` | Must hold the image plus ~50 GB of weights. |
| `workersMin` | `0` | Scale to zero between jobs. |

## 4. Verify, cheapest check first

```bash
EP=<endpoint-id>
# 1. Health check — returns before touching ComfyUI, so it proves only that the
#    worker booted and the handler is registered.
curl -sS -X POST "https://api.runpod.ai/v2/$EP/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"health_check":true}}'

# 2. Text-to-video. A cold start downloads ~50 GB from the mirror first, so submit
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

- R2 bills for storage; egress to the workers is free.
- `workersMin: 0` means no GPU cost while idle, but every cold start pays GPU time for
  the ~50 GB pull. If cold starts are frequent, `workersMin: 1` may well be cheaper —
  measure before assuming either way.
- Delete an endpoint and its template when finished. The R2 bucket is independent of
  Runpod and outlives them.
