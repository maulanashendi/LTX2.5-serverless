#!/usr/bin/env bash

set -euo pipefail

ltx_log() {
    echo "worker-comfyui: $*"
}

ltx_hf_token() {
    printf '%s\n' "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGINGFACE_ACCESS_TOKEN:-}}}"
}

ltx_download_with_wget() {
    local url="$1"
    local output_path="$2"
    local token="${3:-}"
    local tmp_path="${output_path}.part"

    ltx_log "Downloading ${url##*/} via wget"
    if [ -n "${token}" ]; then
        wget -nv -c --header="Authorization: Bearer ${token}" -O "${tmp_path}" "${url}"
    else
        wget -nv -c -O "${tmp_path}" "${url}"
    fi
    mv "${tmp_path}" "${output_path}"
}

ltx_download_with_hf_hub() {
    local url="$1"
    local output_path="$2"
    local token="${3:-}"

    LTX_DOWNLOAD_URL="${url}" \
    LTX_DOWNLOAD_OUTPUT_PATH="${output_path}" \
    LTX_DOWNLOAD_TOKEN="${token}" \
    python - <<'PY'
import os
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

url = os.environ["LTX_DOWNLOAD_URL"]
output_path = Path(os.environ["LTX_DOWNLOAD_OUTPUT_PATH"])
match = re.match(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$", url)
if not match:
    raise SystemExit(f"Unsupported Hugging Face resolve URL: {url}")

repo_id, revision, filename = match.groups()
downloaded_path = Path(
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=os.environ.get("LTX_DOWNLOAD_TOKEN") or None,
        local_dir=str(output_path.parent),
    )
)
if downloaded_path.resolve() != output_path.resolve():
    downloaded_path.replace(output_path)
PY
}

ltx_download() {
    local url="$1"
    local output_path="$2"
    local token="${3:-}"
    local backend="${LTX25_DOWNLOAD_BACKEND:-auto}"

    mkdir -p "$(dirname "${output_path}")"
    if [ -f "${output_path}" ]; then
        ltx_log "LTX asset already present: ${output_path}"
        return
    fi

    case "${backend}" in
        auto)
            if [[ "${url}" == https://huggingface.co/* ]] && python -c "import huggingface_hub" >/dev/null 2>&1; then
                ltx_download_with_hf_hub "${url}" "${output_path}" "${token}"
            else
                ltx_download_with_wget "${url}" "${output_path}" "${token}"
            fi
            ;;
        hf_hub)
            ltx_download_with_hf_hub "${url}" "${output_path}" "${token}"
            ;;
        wget)
            ltx_download_with_wget "${url}" "${output_path}" "${token}"
            ;;
        *)
            ltx_log "Unsupported LTX25_DOWNLOAD_BACKEND='${backend}'"
            exit 1
            ;;
    esac
}

ltx_download_batch() {
    local token="$1"
    shift
    local concurrency="${LTX25_DOWNLOAD_CONCURRENCY:-4}"
    local -a pids=()
    local -a labels=()
    local -a failed_files=()
    local entry url output_path idx

    for entry in "$@"; do
        url="${entry%%|*}"
        output_path="${entry#*|}"

        ltx_download "${url}" "${output_path}" "${token}" &
        pids+=("$!")
        labels+=("${output_path}")

        if [ "${#pids[@]}" -ge "${concurrency}" ]; then
            for idx in "${!pids[@]}"; do
                wait "${pids[${idx}]}" || failed_files+=("${labels[${idx}]}")
            done
            pids=()
            labels=()
        fi
    done

    if [ "${#pids[@]}" -gt 0 ]; then
        for idx in "${!pids[@]}"; do
            wait "${pids[${idx}]}" || failed_files+=("${labels[${idx}]}")
        done
    fi

    if [ "${#failed_files[@]}" -gt 0 ]; then
        ltx_log "Failed to download: ${failed_files[*]}"
        exit 1
    fi
}

ltx_transformer_filename() {
    case "$1" in
        distilled-int8)
            printf '%s\n' "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
            ;;
        *)
            return 1
            ;;
    esac
}

ltx_text_encoder_filename() {
    case "$1" in
        distilled-int8)
            printf '%s\n' "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
            ;;
        *)
            return 1
            ;;
    esac
}

bootstrap_ltx25() {
    local variant="${LTX25_PRELOAD_VARIANT:-}"
    [ -n "${variant}" ] || return

    local model_root="${COMFY_MODEL_ROOT:-/comfyui/models}"
    local repo_url="${LTX25_REPO_URL:-https://huggingface.co/Lightricks/LTX-2.5/resolve/main}"
    local token transformer text_encoder
    token="$(ltx_hf_token)"
    transformer="$(ltx_transformer_filename "${variant}")" || {
        ltx_log "Unsupported LTX25_PRELOAD_VARIANT='${variant}'"
        exit 1
    }
    text_encoder="$(ltx_text_encoder_filename "${variant}")"

    local -a downloads=(
        "${repo_url}/diffusion_models/${transformer}|${model_root}/diffusion_models/${transformer}"
        "${repo_url}/text_encoders/${text_encoder}|${model_root}/text_encoders/${text_encoder}"
        "${repo_url}/vae/ltx-2.5-video-vae-bf16.safetensors|${model_root}/vae/ltx-2.5-video-vae-bf16.safetensors"
        "${repo_url}/vae/ltx-2.5-audio-vae-bf16.safetensors|${model_root}/vae/ltx-2.5-audio-vae-bf16.safetensors"
        "${repo_url}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|${model_root}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    )

    if [ "${LTX25_PRELOAD_PROMPT_ENHANCER:-true}" = "true" ]; then
        local prompt_enhancer="gemma4_e2b_it_bf16.safetensors"
        local enhancer_url="${LTX25_ENHANCER_URL:-https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/${prompt_enhancer}}"
        downloads+=("${enhancer_url}|${model_root}/text_encoders/${prompt_enhancer}")
    fi

    ltx_download_batch "${token}" "${downloads[@]}"

    ltx_log "LTX 2.5 ${variant} model stack is ready"
}
