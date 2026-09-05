# bc250-notebook image

This is the image that JupyterHub spawns for every user. Fedora 44 base
+ Mesa/rusticl OpenCL stack + torch 2.4.0 CPU + patched `pytorch_ocl`
(the `CUSTOM_REDUCE=1` build from `../../../pytorch-dlprim-fix/`) +
`jupyterhub` (for the `jupyterhub-singleuser` launcher the Hub starts in
each pod).

Current tag: **`bc250-notebook:latest`**. Referenced by
`../30-configmap.yaml.tmpl` (`c.KubeSpawner.image`), with
`imagePullPolicy: Never`.

Fedora base, not the upstream Jupyter image, because rusticl on gfx1013 needs
recent Mesa and the container's Mesa userspace has to line up with the host
kernel's `amdgpu`. Same distro/version = no mismatch.

## Build + load into k3s

    ./build.sh

Builds `bc250-notebook:latest` with Docker, then imports it into k3s's
containerd (they are separate image stores). Re-run after any change here,
then `sudo kubectl -n jupyterhub rollout restart deploy/hub` and start a fresh
server so the new image is pulled into the spawned pod.

## Smoke test outside Kubernetes first

    sudo docker run --rm -it \
      --device /dev/dri/renderD128 --group-add 105 \
      bc250-notebook:latest \
      bash -lc 'clinfo -l && python -c "import torch,pytorch_ocl; \
        x=torch.rand(1024,1024,device=\"ocl:0\"); print((x@x).sum())"'

Expect `clinfo -l` to list `Platform #0: rusticl` with the BC-250 device, and
the tensor op to print a number without a compile error.

## Consumed by

- `../30-configmap.yaml.tmpl` — `c.KubeSpawner.image`, the image for every spawned
  user pod (`devic.es/dri` request + `supplemental_gids: [105]` come from the
  spawner config there, added in Stage 4)

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
