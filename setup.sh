#!/usr/bin/env bash
# ============================================================================
# Setup: create a conda env with the same library versions LLaMA-Factory's
# Docker image is built on. That's the path of least resistance.
#
# Reference image hiyouga/llamafactory:latest
#   Ubuntu 22.04 | CUDA 12.4 | Python 3.11 | PyTorch 2.6.0 | DeepSpeed 0.16.5
#
# We pin torch 2.6.0 + the cu121 wheel index, which is forward-compatible
# with any NVIDIA driver supporting CUDA >= 12.1 (so CUDA 12.4 drivers like
# yours work fine). We do NOT chase the newest torch — this matches what
# llamafactory tests against.
#
# Usage:
#   bash setup.sh
#   conda activate <name>
#   bash pipeline.sh
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- Pinned versions (match llamafactory Docker) ---------------------------
PY_VER="3.11"
TORCH_VER="2.6.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"   # cu121 is FC with 12.x drivers
DEEPSPEED_VER="0.16.5"

echo "=========================================="
echo "  LCR Pipeline — Setup"
echo "=========================================="
echo "Repo root: $REPO_ROOT"
echo ""
echo "Pinned versions (match llamafactory Docker image):"
echo "  python:    $PY_VER"
echo "  torch:     $TORCH_VER  (index: $TORCH_INDEX)"
echo "  deepspeed: $DEEPSPEED_VER"
echo ""

# ---- Conda check ------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH." >&2
    echo "  Install Miniconda or Anaconda first: https://docs.conda.io/" >&2
    exit 1
fi
CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "Conda: $(conda --version)  ($CONDA_BASE)"
echo ""

# ---- Driver sanity (informational only) ------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPUs:"
    nvidia-smi -L | sed 's/^/  /'
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | awk '{print $3}' | head -n1 || true)
    if [ -n "$DRIVER_CUDA" ]; then
        echo "  driver supports CUDA up to: $DRIVER_CUDA"
        # cu121 wheels need driver supporting CUDA >= 12.1.
        major=$(echo "$DRIVER_CUDA" | cut -d. -f1)
        minor=$(echo "$DRIVER_CUDA" | cut -d. -f2)
        ver=$((major * 100 + minor))
        if [ "$ver" -lt 1201 ]; then
            echo "  WARNING: your driver may not support CUDA 12.1 wheels." >&2
            echo "           If torch fails to import, your driver needs an update." >&2
        fi
    fi
else
    echo "WARNING: nvidia-smi not found. No GPU detected; setup will still proceed" >&2
    echo "         but pipeline.sh stages 1, 4, 5 require GPUs." >&2
fi
echo ""

# ---- Step 1: env name -------------------------------------------------------
echo "Step 1: environment name"
echo "-------------------------"
read -r -p "  Conda env name [lcr]: " ENV_NAME
ENV_NAME="${ENV_NAME:-lcr}"
ENV_NAME=$(echo "$ENV_NAME" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
[ -z "$ENV_NAME" ] && ENV_NAME="lcr"
echo "  env name: $ENV_NAME"
echo ""

# ---- Step 2: create env -----------------------------------------------------
echo "Step 2: create or update env"
echo "-----------------------------"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "  env '$ENV_NAME' already exists"
    read -p "  Recreate from scratch? (y/N): " RECREATE
    if [[ "${RECREATE:-n}" =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME" -y
        conda create -n "$ENV_NAME" "python=$PY_VER" -y
    fi
else
    conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi
conda activate "$ENV_NAME"
echo "  active: $CONDA_DEFAULT_ENV"
echo "  python: $(python --version)  ($(which python))"
echo ""

# ---- Step 3: install torch (pinned) ----------------------------------------
echo "Step 3: install torch $TORCH_VER (cu121)"
echo "-----------------------------------------"
python -m pip install --upgrade pip wheel setuptools >/dev/null
python -m pip install "torch==$TORCH_VER" --index-url "$TORCH_INDEX"
TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)")
echo "  installed: torch $TORCH_INSTALLED"
echo ""

# ---- Step 4: install deepspeed (pinned) ------------------------------------
echo "Step 4: install deepspeed $DEEPSPEED_VER"
echo "-----------------------------------------"
python -m pip install "deepspeed==$DEEPSPEED_VER"
echo "  done"
echo ""

# ---- Step 5: install remaining requirements --------------------------------
echo "Step 5: install remaining requirements"
echo "---------------------------------------"
python -m pip install -r "$REPO_ROOT/requirements.txt"
echo "  done"
echo ""

# ---- Step 6: LLaMA-Factory --------------------------------------------------
echo "Step 6: LLaMA-Factory"
echo "----------------------"
LF_DIR="${LF_DIR:-$REPO_ROOT/third_party/LLaMA-Factory}"
if [ -d "$LF_DIR/.git" ]; then
    echo "  already cloned: $LF_DIR"
    read -p "  Pull latest? (y/N): " PULL
    if [[ "${PULL:-n}" =~ ^[Yy]$ ]]; then
        (cd "$LF_DIR" && git pull --ff-only)
    fi
else
    mkdir -p "$(dirname "$LF_DIR")"
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
    echo "  cloned: $LF_DIR"
fi
# Install WITHOUT [torch] and [deepspeed] extras since we already pinned both.
(cd "$LF_DIR" && python -m pip install -e ".[metrics]")
echo "  installed editable (excluding [torch] and [deepspeed] to preserve pins)"
echo ""

# ---- Step 7: smoke test -----------------------------------------------------
echo "Step 7: smoke test"
echo "-------------------"
SMOKE_RC=0
python - <<'PY' || SMOKE_RC=$?
import shutil, sys

print("  [import]")
mods = ["torch", "transformers", "datasets", "vllm",
        "deepspeed", "llamafactory", "lm_eval", "numpy", "tqdm"]
missing = []
for m in mods:
    try:
        __import__(m)
        print(f"    OK   {m}")
    except Exception as e:
        print(f"    FAIL {m}  ({e.__class__.__name__}: {e})")
        missing.append(m)
if missing:
    print("  missing imports:", missing)
    sys.exit(1)

print("  [torch CUDA]")
import torch
print(f"    torch.__version__:         {torch.__version__}")
print(f"    torch.version.cuda:        {torch.version.cuda}")
print(f"    torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"    torch.cuda.device_count(): {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        print(f"      GPU {i}: {name}  (sm_{major}{minor})")
    try:
        x = torch.randn(64, 64, device="cuda:0")
        y = x @ x.T
        _ = y.sum().item()
        print("    GPU matmul:                OK")
    except Exception as e:
        print(f"    GPU matmul:                FAIL  ({e.__class__.__name__}: {e})")
        sys.exit(2)
else:
    print("  WARNING: CUDA not available. Pipeline will not be usable for training.")

print("  [CLIs]")
for cli in ("llamafactory-cli", "lm_eval"):
    path = shutil.which(cli)
    if path:
        print(f"    OK   {cli}  ({path})")
    else:
        print(f"    FAIL {cli}  not on PATH")
        sys.exit(3)

print("  smoke test passed.")
PY

case "$SMOKE_RC" in
    0) ;;
    1) echo "  ERROR: required imports failed." >&2; exit 1 ;;
    2) echo "  ERROR: CUDA tensor op failed. Driver too old for cu121 wheels." >&2; exit 1 ;;
    3) echo "  ERROR: required CLI not found." >&2; exit 1 ;;
    *) echo "  ERROR: smoke test failed (rc=$SMOKE_RC)." >&2; exit 1 ;;
esac
echo ""

echo "=========================================="
echo "Setup complete."
echo ""
echo "    conda activate $ENV_NAME"
echo "    bash pipeline.sh"
echo "=========================================="
