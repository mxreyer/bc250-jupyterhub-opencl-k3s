# Notebook image for the BC-250.
#
# Base is Fedora 44 on purpose: the OpenCL path (Mesa rusticl on radeonsi) is
# recent-Mesa-only, and building it here from the SAME distro/version as the
# host (Mesa 26.1.8) sidesteps every library-mismatch problem. The host kernel
# driver (amdgpu) is what the container's Mesa talks to through
# /dev/dri/renderD128, so the userspace versions need to line up.
FROM registry.fedoraproject.org/fedora:44

# OpenCL userspace:
#   mesa-libOpenCL   - rusticl (libRusticlOpenCL.so) + its ICD registration
#   mesa-dri-drivers - the radeonsi gallium driver rusticl runs on top of
#   libclc           - OpenCL built-in function library rusticl needs
#   ocl-icd          - the ICD loader clinfo / pytorch_ocl link against
#   clinfo           - so we can check the device from inside the container
RUN dnf -y install --setopt=install_weak_deps=False \
        mesa-libOpenCL mesa-dri-drivers libclc ocl-icd clinfo \
        python3.12 python3.12-devel \
        procps-ng findutils which tar gzip \
 && dnf clean all

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

# Swap the stock pt_ocl.so for the patched (CUSTOM_REDUCE=1) rebuild.
# build.sh copies this file into the build context first.
COPY pt_ocl.so /tmp/pt_ocl.so
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
