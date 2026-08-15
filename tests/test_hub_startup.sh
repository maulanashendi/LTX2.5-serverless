#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d)"
capture_file="${test_dir}/python-args"
trap 'rm -rf "${test_dir}"' EXIT

mkdir -p "${test_dir}/bin"
cat > "${test_dir}/bin/python" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" > "${HUB_TEST_CAPTURE}"
EOF
chmod +x "${test_dir}/bin/python"

HUB_TEST_CAPTURE="${capture_file}" \
RUNPOD_HUB_VALIDATION=true \
PATH="${test_dir}/bin:${PATH}" \
bash "${repo_root}/src/start.sh"

if [ "$(cat "${capture_file}")" != "-u /handler.py" ]; then
    echo "Hub validation did not start only the RunPod handler" >&2
    exit 1
fi

echo "Hub startup fast path passed"
