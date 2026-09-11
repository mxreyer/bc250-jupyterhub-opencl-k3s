# bc250-notebook image

This is the image that JupyterHub spawns for every user. Fedora 45 base
(`ARG FEDORA_VERSION`, see the Dockerfile header for why and for the 44 vs 45
measurements) + Mesa/rusticl OpenCL stack + Mesa's patched libclc (drop-in over Fedora's,
see below) + torch 2.4.0 CPU + patched `pytorch_ocl` (`pt_ocl.so` from the
releases of [pytorch-dlprim-gfx1013](https://github.com/mxreyer/pytorch-dlprim-gfx1013), fetched by URL + sha256 in the
Dockerfile) + `jupyterhub` (for the
`jupyterhub-singleuser` launcher the Hub starts in each pod).

Current tag: **`bc250-notebook:latest`**. Referenced by
`../30-configmap.yaml.tmpl` (`c.KubeSpawner.image`), with
`imagePullPolicy: Never`.

Fedora base, not the upstream Jupyter image, because rusticl on gfx1013 needs
recent Mesa. The container's Fedora does not need to match the host's: the only
shared piece is the `amdgpu` kernel driver, and Mesa 26.2 (Fedora 45) runs fine
on the host's Fedora 44 kernel — measured, see the Dockerfile.

## The libclc swap

The `Dockerfile` replaces the two SPIR-V files Fedora's `libclc-spirv` installs
under `/usr/lib64/clc/` with the ones from Mesa's maintained fork
(https://gitlab.freedesktop.org/karolherbst/mesa-libclc, release 22.1.8.3,
pinned by URL + sha256). Fedora 44's build has an empty
`__clc_flush_denormal_if_not_supported`, which makes every kernel using
`sin`/`cos`/`tan`/`fma`/`remquo` fail to link — in PyTorch, that is
`torch.randn(..., device="ocl:0")`. It is also the source of the
*"Patched Mesa libclc not detected"* warning. Details:
[OPENCL-PERF.md](https://github.com/mxreyer/pytorch-dlprim-gfx1013/blob/main/OPENCL-PERF.md), Finding 8. To bump: change `MESA_LIBCLC_VER` and both checksums
(`curl -sL <url> | sha256sum`), rebuild, smoke-test.

## Build + load into k3s

    ./build.sh

Builds `bc250-notebook:latest` with Docker (`FEDORA_VERSION=44 ./build.sh`
for the previous base), then imports it into k3s's containerd (they are separate
image stores). Re-run after any change here,
then `sudo kubectl -n jupyterhub rollout restart deploy/hub` and start a fresh
server so the new image is pulled into the spawned pod.

## Smoke test outside Kubernetes first

    sudo docker run --rm -it \
      --device /dev/dri/renderD128 --group-add 105 \
      bc250-notebook:latest \
      bash -lc 'clinfo -l && python -c "import torch,pytorch_ocl; \
        x=torch.rand(1024,1024,device=\"ocl:0\"); print((x@x).sum())"'

Expect `clinfo -l` to list `Platform #0: rusticl` with the BC-250 device and
**no** `Rusticl warning` line above it, and the tensor op to print a number
without a compile error. `torch.randn(64, device="ocl:0")` is the one-liner
that catches the libclc problem specifically.

## Optional: fp16 convolutions

`DLPRIM_CONV_FP16=1` in a pod's environment (or `os.environ[...]` at the very
top of a notebook, before the first convolution runs) switches the Winograd
convolution kernels to an fp16 inner loop: ~1.5× on ResNet-style training and
inference at ~0.5% numerical error, accuracy unchanged over 16 epochs. Off by
default; details in [OPENCL-PERF.md](https://github.com/mxreyer/pytorch-dlprim-gfx1013/blob/main/OPENCL-PERF.md), Finding 11. To make it the
default for every user, add it next to `RUSTICL_ENABLE` in
`c.KubeSpawner.environment` in `../30-configmap.yaml.tmpl`.

## Consumed by

- `../30-configmap.yaml.tmpl` — `c.KubeSpawner.image`, the image for every spawned
  user pod (`devic.es/dri` request + `supplemental_gids: [105]` come from the
  spawner config there, added in Stage 4)

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
