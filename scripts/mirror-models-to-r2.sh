#!/usr/bin/env bash
#
# One-shot operator script: mirrors the LTX 2.5 model weights from Hugging
# Face to a Cloudflare R2 bucket, so bootstrap_ltx25.sh can pull them from R2
# via LTX25_REPO_URL instead of hitting Hugging Face at cold start.
#
# WARNING: Lightricks/LTX-2.5 is a gated Hugging Face repo. Before running
# this script, confirm that its license terms permit mirroring the weights
# into your own bucket. Run this on your own machine, not inside the worker
# image.

set -euo pipefail

mirror_log() {
    echo "mirror-models-to-r2: $*"
}

: "${R2_BUCKET:?R2_BUCKET is required}"
: "${R2_ENDPOINT:?R2_ENDPOINT is required (e.g. https://<account>.r2.cloudflarestorage.com)}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required}"
: "${HUGGINGFACE_ACCESS_TOKEN:?HUGGINGFACE_ACCESS_TOKEN is required}"

variant="${LTX25_PRELOAD_VARIANT:-distilled-int8}"

transformer_filename() {
    case "$1" in
        distilled-int8)
            printf '%s\n' "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
            ;;
        *)
            return 1
            ;;
    esac
}

text_encoder_filename() {
    case "$1" in
        distilled-int8)
            printf '%s\n' "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
            ;;
        *)
            return 1
            ;;
    esac
}

transformer="$(transformer_filename "${variant}")" || {
    mirror_log "Unsupported LTX25_PRELOAD_VARIANT='${variant}'"
    exit 1
}
text_encoder="$(text_encoder_filename "${variant}")"

hf_repo_url="https://huggingface.co/Lightricks/LTX-2.5/resolve/main"
enhancer_repo_url="https://huggingface.co/Comfy-Org/gemma-4/resolve/main"

relative_paths=(
    "diffusion_models/${transformer}"
    "text_encoders/${text_encoder}"
    "vae/ltx-2.5-video-vae-bf16.safetensors"
    "vae/ltx-2.5-audio-vae-bf16.safetensors"
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    "text_encoders/gemma4_e2b_it_bf16.safetensors"
)

source_url_for() {
    case "$1" in
        text_encoders/gemma4_e2b_it_bf16.safetensors)
            printf '%s/%s\n' "${enhancer_repo_url}" "$1"
            ;;
        *)
            printf '%s/%s\n' "${hf_repo_url}" "$1"
            ;;
    esac
}

remote_size() {
    aws s3api head-object \
        --endpoint-url "${R2_ENDPOINT}" \
        --bucket "${R2_BUCKET}" \
        --key "$1" \
        --query 'ContentLength' \
        --output text 2>/dev/null || true
}

# Size according to Hugging Face, so an already-mirrored file can be skipped
# without downloading it first. Follows redirects to the CDN.
source_size() {
    wget --spider --server-response --max-redirect=20 \
        --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        "$1" 2>&1 \
        | awk 'BEGIN{IGNORECASE=1} /^ *Content-Length:/ {n=$2} END{gsub(/\r/,"",n); print n}'
}

# The transformer alone is ~21GB, so this needs real room. Point
# MIRROR_WORK_DIR at a disk that has it; the default /tmp usually does not.
work_dir_base="${MIRROR_WORK_DIR:-${TMPDIR:-/tmp}}"
mkdir -p "${work_dir_base}"
available_kb="$(df -Pk "${work_dir_base}" | awk 'NR==2 {print $4}')"
required_kb=$((25 * 1024 * 1024))
if [ "${available_kb}" -lt "${required_kb}" ]; then
    mirror_log "Only $((available_kb / 1024 / 1024))GB free on ${work_dir_base}; need ~25GB for the largest file."
    mirror_log "Set MIRROR_WORK_DIR to a directory with more room and re-run."
    exit 1
fi

work_dir="$(mktemp -d "${work_dir_base}/ltx25-mirror.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

for relative_path in "${relative_paths[@]}"; do
    dest_key="${relative_path}"
    existing_size="$(remote_size "${dest_key}")"

    local_path="${work_dir}/${relative_path}"
    mkdir -p "$(dirname "${local_path}")"

    source_url="$(source_url_for "${relative_path}")"

    if [ -n "${existing_size}" ] && [ "${existing_size}" != "None" ]; then
        upstream_size="$(source_size "${source_url}")"
        if [ -n "${upstream_size}" ] && [ "${existing_size}" = "${upstream_size}" ]; then
            mirror_log "Skipping ${relative_path}: already in R2 at ${existing_size} bytes"
            continue
        fi
    fi

    mirror_log "Downloading ${relative_path} from Hugging Face"
    wget -nv --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
        -O "${local_path}" "${source_url}"

    local_size="$(stat -c%s "${local_path}" 2>/dev/null || stat -f%z "${local_path}")"

    mirror_log "Uploading ${relative_path} to R2"
    aws s3 cp "${local_path}" "s3://${R2_BUCKET}/${dest_key}" \
        --endpoint-url "${R2_ENDPOINT}"

    rm -f "${local_path}"
done

mirror_log "Mirror complete. Set the following on the RunPod endpoint template:"
mirror_log "LTX25_REPO_URL=${R2_ENDPOINT}/${R2_BUCKET}"
