# JupyterHub with GPU-accelerated PyTorch on BC-250

- Serve **GPU-accelerated PyTorch notebooks** from an AsRock **BC-250**
  mining board: JupyterHub on a single-node Kubernetes (k3s) cluster on the
  same machine, one GPU-enabled pod and persistent disk per user.
- The GPU is handed to **unprivileged** pods by a device plugin.
- Remote access and HTTPS is handled via **Tailscale** as an in-pod sidecar.
- **Learn Kubernetes along the way.** This repo contains a walkthrough
  plus every manifest, Dockerfile, and helper script. It is explicitly for
  learning, **not a production reference**.

This repo is only the Kubernetes / JupyterHub half of the project. **Getting PyTorch onto the BC-250's GPU in the first place** is described in [**pytorch-gpu-bc250**](https://github.com/mxreyer/pytorch-gpu-bc250), alongside benchmarking results against cloud GPUs.

There is a **write-up** of the whole project: [Turning a mining board into a multi-user ML box](https://mxreyer.github.io/blog/bc250-ml-box/)
  ([source](https://github.com/mxreyer/blog)).

## Architecture, end to end

```
                    AsRock BC-250  ·  Fedora 44  ·  amdgpu  ·  Mesa rusticl on radeonsi
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                       │
 │   LAN  ──HTTP──▶  Traefik (:80)  ──Ingress──▶  Service  ──┐                           │
 │   (sslip.io)                                              │                           │
 │                                                           ▼                           │
 │                                                ┌──────────────────────────┐           │
 │   tailnet ──HTTPS(:443)──▶  tailscale  ──────▶  hub pod                   │           │
 │   (Serve, real cert)        sidecar            │   jupyterhub + proxy     │           │
 │                          (in the hub pod)      │   + idle-culler          │           │
 │                                                │   KubeSpawner ──────┐    │           │
 │                                                └─────────────────────┼────┘           │
 │                                 ┌────────────────────────────────────┘                │
 │                                 │   via RBAC (Role, namespaced)                       │
 │                                 ▼                                                     │
 │                     ┌────────────────────────┐                                        │
 │                     │  jupyter-<user> pod    │ (one per logged-in user)               │
 │                     │  PyTorch on ocl:0      │                                        │
 │                     │  PVC  claim-<user>     │                                        │
 │                     │  requests devic.es/dri │                                        │
 │                     └───────────┬────────────┘                                        │
 │                                 ▼                                                     │
 │                   generic-device-plugin (DaemonSet)                                   │
 │                                 ▼                                                     │
 │                   /dev/dri/renderD128  ──▶  GPU (gfx1013)                             │
 │                                                                                       │
 │   k3s bundled: CoreDNS · Traefik · local-path provisioner · metrics-server            │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

- **LAN** clients hit Traefik over HTTP at `*.<LAN-IP>.sslip.io`.
- **Tailnet** clients hit the in-pod Tailscale sidecar over HTTPS; it
  terminates TLS and forwards to the Hub on loopback.
- The **Hub** spawns a **per-user pod + PVC** on login.
- Each per-user pod **requests the GPU** as `devic.es/dri`, handed over by the
  **device plugin**; the pods themselves are unprivileged.
- **PyTorch** talks to the GPU as `ocl:0` via `pytorch_ocl` → rusticl →
  radeonsi → `amdgpu`.

## OpenCL, not ROCm

gfx1013 is not an officially supported ROCm target, so PyTorch reaches the
GPU through **rusticl** (Mesa's OpenCL implementation) and a patched
[`pytorch_ocl`](https://github.com/artyom-beilis/pytorch_dlprim); how and
why is in [pytorch-gpu-bc250](https://github.com/mxreyer/pytorch-gpu-bc250). For the
Kubernetes side this is a simplification: the OpenCL path needs only the DRI
render node, so exposing the GPU to a pod is a plain device request.

## Prerequisites

- **AsRock BC-250** with PyTorch working on its GPU on bare metal:
  [pytorch-gpu-bc250](https://github.com/mxreyer/pytorch-gpu-bc250), Stages 1-2
  (Fedora, Mesa ≥ 26.1, rusticl, the patched `pt_ocl.so`).
- **Docker** and **k3s** (single-node) on the box.
- Optional, for remote access: **Tailscale**.

## Setup

Full detail in [SETUP.md](SETUP.md). Stages:

| Stage | Result |
| ----- | ------ |
| 0 | *(prerequisite, [pytorch-gpu-bc250](https://github.com/mxreyer/pytorch-gpu-bc250))* PyTorch runs on the GPU on bare metal |
| 1 | JupyterHub on k3s: CPU notebook pods with per-user persistent disks and idle culling, reachable on the LAN and over Tailscale |
| 2 | Those per-user pods use the GPU, as unprivileged pods |

## Notes

- Real Kubernetes `Secret` objects are never committed. Files named
  `*.example.txt` are templates that document the keys, nothing more.
- Host-specific values (LAN IP, Tailscale hostname / tailnet / tag,
  admin username) are not in the manifests. They live in the gitignored
  `k8s/env.local`, created from `k8s/env.example`. `k8s/render.sh` writes
  the real `k8s/**/<name>.yaml` from the committed `<name>.yaml.tmpl`
  files. **Run `./k8s/render.sh` after any edit to `env.local` or a
  template.**

## Repository layout

| Path | What it is |
| --- | --- |
| [SETUP.md](SETUP.md) | End-to-end walkthrough. Stages 1-2. |
| `k8s/env.example`, `k8s/render.sh` | Host-specific values + the renderer that turns `k8s/**/*.yaml.tmpl` into manifests you can apply. Run `./k8s/render.sh` before the first apply. |
| `k8s/device-plugin/` | `generic-device-plugin` DaemonSet — advertises `devic.es/dri`. Required for any GPU pod (Stage 2). |
| `k8s/jupyterhub/` | The whole stack (namespace `jupyterhub`): RBAC, ConfigMap, PVC, Deployment (hub + Tailscale sidecar), Service, Ingress. |
| `k8s/jupyterhub/image/` | The Hub image: JupyterHub + `kubespawner` + `idle-culler` + `nativeauthenticator`. |
| `k8s/jupyterhub/notebook-image/` | The `bc250-notebook` image the Hub spawns per user: Fedora + rusticl + torch + patched `pytorch_ocl`. |
| `k8s/jupyterhub/gpu-check.py` | Proves a running user pod is on the GPU, not silently on the CPU (Stage 2). |

## Related repositories

| Repository | What it is |
| --- | --- |
| [pytorch-gpu-bc250](https://github.com/mxreyer/pytorch-gpu-bc250) | Getting PyTorch onto the BC-250's GPU on bare metal, plus the ResNet-9 / CIFAR-10 benchmark against cloud GPUs (harness, protocol, every result). |
| [pytorch-dlprim-gfx1013](https://github.com/mxreyer/pytorch-dlprim-gfx1013) | The `pytorch_ocl` patches and build script, per-patch benchmark numbers, microbenchmarks and correctness sweeps. The notebook image fetches its `pt_ocl.so` release. |
| [blog](https://github.com/mxreyer/blog) | The write-up of the whole project, [published here](https://mxreyer.github.io/blog/bc250-ml-box/). |

## License

[MIT](LICENSE) — Copyright (c) 2026 mxreyer.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
