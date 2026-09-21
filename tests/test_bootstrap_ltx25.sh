#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_TO_TEST="${REPO_ROOT}/src/bootstrap_ltx25.sh"

TEST_DIR="$(mktemp -d)"
trap 'rm -rf "${TEST_DIR}"' EXIT

BIN_DIR="${TEST_DIR}/bin"
WGET_LOG_FILE="${TEST_DIR}/wget.log"
MODEL_ROOT="${TEST_DIR}/models"
mkdir -p "${BIN_DIR}"

cat > "${BIN_DIR}/wget" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output_path=""
url=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -O) output_path="$2"; shift 2 ;;
        --header=*) printf 'header:%s\n' "${1#--header=}" >> "${WGET_LOG_FILE}"; shift ;;
        -nv|-c|-q) shift ;;
        *) url="$1"; shift ;;
    esac
done
[ -n "${output_path}" ] && [ -n "${url}" ]
mkdir -p "$(dirname "${output_path}")"
printf 'url:%s\nout:%s\n' "${url}" "${output_path}" >> "${WGET_LOG_FILE}"
printf 'downloaded:%s\n' "${url}" > "${output_path}"
EOF
chmod +x "${BIN_DIR}/wget"

(
    export PATH="${BIN_DIR}:${PATH}"
    export WGET_LOG_FILE
    export COMFY_MODEL_ROOT="${MODEL_ROOT}"
    export LTX25_DOWNLOAD_BACKEND="wget"
    export HUGGINGFACE_ACCESS_TOKEN="hf-test-token"
    export LTX25_PRELOAD_VARIANT="distilled-int8"
    export LTX25_PRELOAD_PROMPT_ENHANCER=true
    source "${SCRIPT_TO_TEST}"
    bootstrap_ltx25
    bootstrap_ltx25
)

for expected_file in \
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" \
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" \
    "text_encoders/gemma4_e2b_it_bf16.safetensors" \
    "vae/ltx-2.5-video-vae-bf16.safetensors" \
    "vae/ltx-2.5-audio-vae-bf16.safetensors" \
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"; do
    [ -f "${MODEL_ROOT}/${expected_file}" ] || {
        echo "Expected ${MODEL_ROOT}/${expected_file} to exist"
        exit 1
    }
done

[ "$(grep -c '^url:' "${WGET_LOG_FILE}")" -eq 6 ]
[ "$(grep -c '^header:Authorization: Bearer hf-test-token$' "${WGET_LOG_FILE}")" -eq 6 ]
# Five of the six assets come from the LTX repo; the prompt enhancer lives in a
# different HuggingFace repo (Comfy-Org/gemma-4) and has its own default URL.
[ "$(grep -c '^url:https://huggingface\.co/Lightricks/LTX-2\.5/resolve/main/' "${WGET_LOG_FILE}")" -eq 5 ]
[ "$(grep -c '^url:https://huggingface\.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e2b_it_bf16\.safetensors$' "${WGET_LOG_FILE}")" -eq 1 ]

if (
    export LTX25_PRELOAD_VARIANT="made-up-variant"
    source "${SCRIPT_TO_TEST}"
    bootstrap_ltx25 >/dev/null 2>&1
); then
    echo "Expected unsupported LTX variant to fail"
    exit 1
fi

# LTX25_REPO_URL overrides the base URL, keeps the relative layout, and
# downloading without a token (as a public R2 bucket would be) does not fail
# or send an Authorization header.
CUSTOM_REPO_LOG="${TEST_DIR}/wget-custom-repo.log"
CUSTOM_MODEL_ROOT="${TEST_DIR}/models-custom-repo"
(
    unset HF_TOKEN HUGGINGFACE_TOKEN HUGGINGFACE_ACCESS_TOKEN
    export PATH="${BIN_DIR}:${PATH}"
    export WGET_LOG_FILE="${CUSTOM_REPO_LOG}"
    export COMFY_MODEL_ROOT="${CUSTOM_MODEL_ROOT}"
    export LTX25_DOWNLOAD_BACKEND="wget"
    export LTX25_REPO_URL="https://r2.example.com/ltx25-weights"
    export LTX25_PRELOAD_VARIANT="distilled-int8"
    export LTX25_PRELOAD_PROMPT_ENHANCER=false
    source "${SCRIPT_TO_TEST}"
    bootstrap_ltx25
)

[ -f "${CUSTOM_REPO_LOG}" ] || {
    echo "Expected ${CUSTOM_REPO_LOG} to exist"
    exit 1
}
[ "$(grep -c '^url:https://r2\.example\.com/ltx25-weights/' "${CUSTOM_REPO_LOG}")" -eq 5 ]
if grep -q 'huggingface\.co' "${CUSTOM_REPO_LOG}"; then
    echo "Did not expect a huggingface.co URL when LTX25_REPO_URL is overridden"
    exit 1
fi
if grep -q '^header:Authorization' "${CUSTOM_REPO_LOG}"; then
    echo "Did not expect an Authorization header when no token is set"
    exit 1
fi
[ -f "${CUSTOM_MODEL_ROOT}/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" ] || {
    echo "Expected the transformer to be downloaded from the custom repo URL"
    exit 1
}

# A single failed parallel download must fail the whole bootstrap, not get
# swallowed by `wait`.
FAIL_BIN_DIR="${TEST_DIR}/bin-fail"
mkdir -p "${FAIL_BIN_DIR}"
cat > "${FAIL_BIN_DIR}/wget" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output_path=""
url=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -O) output_path="$2"; shift 2 ;;
        --header=*) shift ;;
        -nv|-c|-q) shift ;;
        *) url="$1"; shift ;;
    esac
done
[ -n "${output_path}" ] && [ -n "${url}" ]
case "${url}" in
    *gemma4-12b-with-proj*)
        echo "simulated network failure" >&2
        exit 1
        ;;
esac
mkdir -p "$(dirname "${output_path}")"
printf 'downloaded:%s\n' "${url}" > "${output_path}"
EOF
chmod +x "${FAIL_BIN_DIR}/wget"

if (
    export PATH="${FAIL_BIN_DIR}:${PATH}"
    export COMFY_MODEL_ROOT="${TEST_DIR}/models-fail"
    export LTX25_DOWNLOAD_BACKEND="wget"
    export HUGGINGFACE_ACCESS_TOKEN="hf-test-token"
    export LTX25_PRELOAD_VARIANT="distilled-int8"
    export LTX25_PRELOAD_PROMPT_ENHANCER=false
    source "${SCRIPT_TO_TEST}"
    bootstrap_ltx25 >/dev/null 2>&1
); then
    echo "Expected a failed parallel download to make bootstrap_ltx25 exit non-zero"
    exit 1
fi

echo "✅ bootstrap_ltx25 preload behavior verified"
