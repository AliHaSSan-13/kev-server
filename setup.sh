#!/usr/bin/env bash
#
# One-command setup and start for the Kev decision API.
#
#   git clone https://github.com/AliHaSSan-13/kev-server-colab.git
#   cd kev-server-colab
#   bash setup.sh
#
# Works the same in the Colab terminal and on a local machine. The script clones
# kev, copies this project's files into the kev root and runs everything from
# there, because `import kev` needs the kev package on the path, which is only
# true inside a kev checkout.
#
# It also works if you already copied main.py / requirements.txt / this script
# into a kev clone by hand: the clone and copy steps are then skipped.
#
# Optional environment variables:
#   KEV_MODEL    kev-0.8b (default) or kev-4b
#   KEV_DEVICE   cuda or cpu (default: cuda when a GPU is present)
#   KEV_HOST     default 127.0.0.1 (use 0.0.0.0 to accept non-tunnelled traffic)
#   KEV_PORT     default 8000
#   KEV_API_KEY  your API key; if unset, main.py generates and saves one
#   KEV_UPSTREAM override the kev clone URL
#   KEV_DIR      directory to clone kev into (default: kev)
#
# The script is idempotent: re-running it skips the clone and the model download.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

KEV_MODEL="${KEV_MODEL:-kev-0.8b}"
KEV_REPO="jaredpalmer/${KEV_MODEL}"
KEV_UPSTREAM="${KEV_UPSTREAM:-https://github.com/jaredpalmer/kev.git}"
KEV_DIR="${KEV_DIR:-kev}"

PY="$(command -v python3 || command -v python || true)"

if [ -z "$PY" ]; then
    echo >&2
    echo "error: python 3 not found on PATH (kev needs 3.12 or 3.13)" >&2
    exit 1
fi

echo "=============================================="
echo "Kev decision API - setup"
echo "=============================================="
echo "Model:  ${KEV_REPO}"
echo "Python: ${PY}"
echo "Source: ${REPO_DIR}"

# ---------------------------------------------------------------- environment
# Never fatal: on a clean machine torch may not be installed yet, and
# requirements.txt installs it a few steps down.
"$PY" - <<'PYCODE'
import sys

print(f"Version: {sys.version.split()[0]}")

try:
    import torch
except ImportError:
    print("torch:   not installed yet - requirements.txt will install it")
else:
    print(f"torch:   {torch.__version__}")

    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    else:
        print("GPU:     none detected - this will run on CPU and be very slow.")
        print("         In Colab: Runtime > Change runtime type > GPU.")
PYCODE

# ----------------------------------------------------------------------- kev
if [ -f pyproject.toml ] && [ -d kev ]; then
    # already sitting in a kev checkout
    echo "==> using the kev checkout at $(pwd)"
else
    if [ -d "${KEV_DIR}/.git" ]; then
        echo "==> ${KEV_DIR} already cloned"
    else
        echo "==> cloning ${KEV_UPSTREAM} into ${KEV_DIR}"
        git clone --depth 1 "${KEV_UPSTREAM}" "${KEV_DIR}"
    fi

    # kev has no main.py and no requirements.txt, so neither copy overwrites
    # anything upstream.
    for file in main.py requirements.txt; do
        if [ ! -f "${REPO_DIR}/${file}" ]; then
            echo >&2
            echo "error: ${file} is missing from ${REPO_DIR}" >&2
            exit 1
        fi
        cp -f "${REPO_DIR}/${file}" "${KEV_DIR}/${file}"
        echo "==> copied ${file} -> ${KEV_DIR}/${file}"
    done

    if [ -f "${REPO_DIR}/README.md" ]; then
        cp -f "${REPO_DIR}/README.md" "${KEV_DIR}/README_COLLAB.md"
        echo "==> copied README.md -> ${KEV_DIR}/README_COLLAB.md"
    fi

    cd "${KEV_DIR}"
fi

# --------------------------------------------------------------------- check
if ! "$PY" -c "import kev.predictors" >/dev/null 2>&1; then
    echo >&2
    echo "error: the kev package is not importable from $(pwd)" >&2
    echo "       python main.py has to run from the root of a kev checkout" >&2
    exit 1
fi

# -------------------------------------------------------------- dependencies
echo "==> installing dependencies"
"$PY" -m pip install -q --upgrade pip

# The upper bound matters: an unbounded `pip install --upgrade huggingface_hub`
# pulls 2.x, which transformers, tokenizers and datasets all refuse to run with.
"$PY" -m pip install -q --upgrade "huggingface_hub>=0.34,<2.0"
"$PY" -m pip install -q -r requirements.txt

# --------------------------------------------------------------------- model
mkdir -p models

if [ -d "models/${KEV_MODEL}" ]; then
    echo "==> models/${KEV_MODEL} already present"
else
    echo "==> downloading ${KEV_REPO}"
    hf download "${KEV_REPO}" --local-dir "models/${KEV_MODEL}"
fi

# The base model (Qwen3.5-0.8B-Base for kev-0.8b) is fetched by kev on the first
# load and cached, so later runs need no network.

# --------------------------------------------------------------------- serve
echo "==> starting the API on http://${KEV_HOST:-127.0.0.1}:${KEV_PORT:-8000}"
exec "$PY" main.py