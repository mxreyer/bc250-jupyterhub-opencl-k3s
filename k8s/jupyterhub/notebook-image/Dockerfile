# Notebook image for the BC-250.
#
# Base is Fedora because the OpenCL path (Mesa rusticl on radeonsi) is
# recent-Mesa-only and Fedora tracks Mesa closely. The only thing the container
# shares with the host is the amdgpu kernel driver behind /dev/dri/renderD128,
# and that interface is stable across Mesa minors, so the container's Fedora
# does not have to match the host's (host: 44). Measured 2026-09-10, both on
# the same host kernel:
#   44 -> Mesa 26.1.8: fma() is a software emulation
#   45 -> Mesa 26.2.2: fma() is the hardware instruction; training and
#         inference within run-to-run noise of 44; correct on both (0 bad / 300
#         sweeps of wino-repro.py with the shipped default).
# Measurements: OPENCL-PERF.md in https://github.com/mxreyer/pytorch-dlprim-gfx1013.
# Override at build time: FEDORA_VERSION=44 ./build.sh
ARG FEDORA_VERSION=45
FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION}

# OpenCL userspace:
#   mesa-libOpenCL   - rusticl (libRusticlOpenCL.so) + its ICD registration
#   mesa-dri-drivers - the radeonsi gallium driver rusticl runs on top of
#   libclc           - OpenCL built-in function library rusticl needs (its
#                      SPIR-V half is replaced with Mesa's fork two steps down)
#   ocl-icd          - the ICD loader clinfo / pytorch_ocl link against
#   clinfo           - so we can check the device from inside the container
RUN dnf -y install --setopt=install_weak_deps=False \
        mesa-libOpenCL mesa-dri-drivers libclc ocl-icd clinfo \
        python3.12 python3.12-devel \
        procps-ng findutils which tar gzip \
 && dnf clean all

# Replace Fedora's libclc SPIR-V library with Mesa's maintained fork.
#
# rusticl loads /usr/lib64/clc/spirv64-mesa3d-.spv (path baked in at Mesa build
# time, no override) and links every kernel against it. Fedora's libclc-spirv
# 22.1.8 (both -1.fc44 and -2.fc45 - byte-identical .spv) has an EMPTY body for
# __clc_flush_denormal_if_not_supported, so any kernel that reaches it fails to
# build with "Internal compilation error: nir_shader not fully linked". That is
# every float sin()/cos()/tan()/sincos() and remquo() (plus fma() on Mesa
# 26.1) - which in PyTorch terms is torch.randn / .normal_() / nn.init.normal_()
# on the device (dlprimitives' Box-Muller uses sin/cos). It is also what prints
# the "Patched Mesa libclc not detected" warning on every run.
#
# Mesa's fork fixes exactly that (commit 60e591d "mark
# __clc_flush_denormal_if_not_supported as static inline") and ships pre-built
# drop-in binaries per release. Same LLVM 22.1.8 that Fedora 44 and 45 ship,
# so nothing else changes. Pinned by URL and sha256; ADD from a URL creates the
# file 0600, hence --chmod. See OPENCL-PERF.md, Finding 8, in
# https://github.com/mxreyer/pytorch-dlprim-gfx1013.
ARG MESA_LIBCLC_VER=22.1.8.3
ARG MESA_LIBCLC_URL=https://gitlab.freedesktop.org/api/v4/projects/27606/packages/generic/mesa-libclc/${MESA_LIBCLC_VER}
ADD --chmod=0644 --checksum=sha256:728cdfa747a1857829c104291984964971783a301a62e9468a3d5a4f1750c949 \
    ${MESA_LIBCLC_URL}/spirv64-mesa3d-.spv /usr/lib64/clc/spirv64-mesa3d-.spv
ADD --chmod=0644 --checksum=sha256:59552f9e22ec221061e9939fc90a46e4159d1400b2e1f85eff46924489d532aa \
    ${MESA_LIBCLC_URL}/spirv-mesa3d-.spv   /usr/lib64/clc/spirv-mesa3d-.spv

# Just an unprivileged user to run the notebook. Access to /dev/dri/renderD128
# is granted at runtime by adding the host's "render" GID (105) as a numeric
# supplemental group - `docker run --group-add 105` for the smoke test,
# securityContext.supplementalGroups: [105] in the k8s Deployment. No named
# group needed in the image (and the Fedora base already ships one called
# "render" at a different GID, which we leave alone).
RUN useradd -m -u 1000 jovyan

ENV VENV=/opt/venv \
    HOME=/home/jovyan \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    RUSTICL_ENABLE=radeonsi \
    DLPRIM_CACHE_DIR=/home/jovyan/.dlprimitives

# Python env mirrors the working host venv exactly: same torch, same
# pytorch_ocl wheel (its stock pt_ocl.so is replaced two steps down).
RUN python3.12 -m venv $VENV \
 && $VENV/bin/pip install --no-cache-dir -U pip \
 && $VENV/bin/pip install --no-cache-dir torch==2.4.0 \
        --index-url https://download.pytorch.org/whl/cpu \
 && $VENV/bin/pip install --no-cache-dir \
        numpy==2.5.2 jupyterlab "jupyterhub>=5,<6" \
        "https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"
# jupyterhub is here for the `jupyterhub-singleuser` launcher: KubeSpawner
# overrides the CMD below with `jupyterhub-singleuser` so the pod registers
# with the Hub and accepts Hub-issued auth. The `jupyter lab` CMD is only a
# fallback for running the image by hand (the smoke test in README.md).

# Swap the stock pt_ocl.so for the patched rebuild from
# https://github.com/mxreyer/pytorch-dlprim-gfx1013: CUSTOM_REDUCE=1 (needed to compile at
# all on rusticl) plus the Winograd performance work (909 -> 2,055 img/s) and
# the half-tensor fixes. Fetched from that repo's GitHub release, pinned by
# URL + sha256 like the libclc above. To bump: new release there, then update
# both ARGs here (`sha256sum pt_ocl.so`) and rebuild.
ARG PT_OCL_VER=v1.0.0
ARG PT_OCL_URL=https://github.com/mxreyer/pytorch-dlprim-gfx1013/releases/download/${PT_OCL_VER}/pt_ocl.so
ARG PT_OCL_SHA256=6c44c6344426caeaaa261db9cbac02ca9bf50515ed45280c485b8d27a4ec9429
ADD --chmod=0644 --checksum=sha256:${PT_OCL_SHA256} ${PT_OCL_URL} /tmp/pt_ocl.so
RUN install -m0644 /tmp/pt_ocl.so \
      "$VENV/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so" \
 && rm /tmp/pt_ocl.so \
 && install -d -o jovyan -g jovyan /home/jovyan/.dlprimitives /home/jovyan/work

USER jovyan
WORKDIR /home/jovyan
EXPOSE 8888

# Fallback CMD for running the image by hand (the smoke test in README.md).
# Under the Hub, KubeSpawner replaces it with `jupyterhub-singleuser` and
# supplies JUPYTERHUB_API_TOKEN via a Secret it manages.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.root_dir=/home/jovyan"]
