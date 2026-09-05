#!/usr/bin/env bash
# Rebuild pytorch_ocl (pytorch_dlprim) with the portable work-group reduction
# forced on, so it doesn't need rusticl's "patched Mesa libclc". See README.md
# in this directory for why.
#
# Self-contained: run from anywhere. It builds a throwaway Python 3.12 venv in
# ./scratch (gitignored) with the SAME torch + pytorch_ocl the notebook image
# uses, applies the patch, and drops the rebuilt pt_ocl.so next to this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$HERE/scratch"
VENV="$SCRATCH/venv"

# Match the notebook image (k8s/jupyterhub/notebook-image/Dockerfile) exactly.
TORCH_VERSION="2.4.0"
PYTORCH_OCL_WHL="https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"

mkdir -p "$SCRATCH"

if [[ -x "$VENV/bin/python3.12" ]]; then
    echo "== reusing scratch venv: $VENV  (rm -rf it to rebuild clean) =="
else
    echo "== building scratch venv: $VENV =="
    python3.12 -m venv "$VENV"
    "$VENV/bin/pip" install -q -U pip
    "$VENV/bin/pip" install -q "torch==$TORCH_VERSION" \
        --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/pip" install -q pybind11 "$PYTORCH_OCL_WHL"
fi

echo "== cloning pytorch_dlprim (with dlprimitives submodule) =="
rm -rf "$SCRATCH/src" "$SCRATCH/build"
git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/artyom-beilis/pytorch_dlprim "$SCRATCH/src"

echo "== applying custom_reduce.patch =="
git -C "$SCRATCH/src/dlprimitives" apply "$HERE/custom_reduce.patch"

echo "== configuring =="
mkdir -p "$SCRATCH/build"
cd "$SCRATCH/build"
cmake \
    -DCMAKE_PREFIX_PATH="$VENV/lib64/python3.12/site-packages/torch/share/cmake/Torch;$VENV/lib64/python3.12/site-packages/pybind11/share/cmake/pybind11" \
    -DPython3_EXECUTABLE="$VENV/bin/python3.12" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    "$SCRATCH/src"

echo "== building =="
# -j4, not -j$(nproc): each pt_ocl .cpp pulls in the full libtorch headers and
# peaks around 1.5 GB of RAM. On the BC-250's 16 GB unified memory a full-width
# parallel build swaps itself to a crawl. Override with JOBS=N if you have more.
make -j"${JOBS:-4}"

echo "== installing pt_ocl.so next to this script =="
install -m0644 "$SCRATCH/build/pytorch_ocl/pt_ocl.so" "$HERE/pt_ocl.so"
strip --strip-unneeded "$HERE/pt_ocl.so"

echo
echo "Built and staged: $HERE/pt_ocl.so"
echo "  - the notebook image picks it up from here (k8s/jupyterhub/notebook-image/build.sh)"
echo "  - to try it in the scratch venv directly:"
echo "      cp '$SCRATCH/build/pytorch_ocl/pt_ocl.so' \\"
echo "         '$VENV/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so'"
echo "      rm -f ~/.dlprimitives/cache.db   # old cache keyed to the old kernel source"
