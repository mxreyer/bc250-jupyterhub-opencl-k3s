# JupyterHub with GPU-accelerated PyTorch on BC-250

- Turn your AsRock **BC-250** mining board into a GPU-accelerated ML
  box (PyTorch through OpenCL) that serves JupyterHub notebooks from a
  single-node Kubernetes (k3s) cluster on the same machine.
- Remote access and HTTPS is handled via **Tailscale** as an in-pod sidecar.
- Compare your box with a cross-platform [**benchmark**](BENCHMARK.md) vs. CUDA cloud GPUs.
- **Learn Kubernetes along the way.** This repo contains a walkthrough
  plus every manifest, Dockerfile, and helper script. It is explicitly for
  learning, **not a production reference**.

## Architecture, end to end

```
                    AsRock BC-250  ·  Fedora 44  ·  amdgpu  ·  Mesa rusticl on radeonsi
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                       │
 │   LAN  ──HTTP──▶  Traefik (:80)  ──Ingress──▶  Service  ──┐                           │
 │   (sslip.io)                                              │                           │
 │                                                           ▼                           │
 │                                              ┌──────────────────────────┐             │
 │   tailnet ──HTTPS(:443)──▶  tailscale  ──────▶  hub pod                 │             │
 │   (Serve, real cert)        sidecar          │   jupyterhub + proxy     │             │
 │                          (in the hub pod)    │   + idle-culler          │             │
 │                                              │   KubeSpawner ──────┐    │             │
 │                                              └─────────────────────┼────┘             │
 │                                 ┌──────────────────────────────────┘                  │
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

gfx1013 is not an officially supported ROCm target. Instead, this build
uses **rusticl** (Mesa's OpenCL implementation). This also simplifies
the Kubernetes stage: the OpenCL path needs only the DRI render node, so
exposing the GPU to a pod is a plain device request.

This uses a patched version of `pytorch_ocl` from
[artyom-beilis/pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim).

Trade-off: Performance well below native CUDA/ROCm. See
[BENCHMARK.md](BENCHMARK.md) — roughly 3× slower than a cloud T4 in
FP32, ~5–6× against a T4 running mixed precision. The point is that it
*works* on hardware ROCm refuses.

## Prerequisites

- **AsRock BC-250**, on Fedora with recent Mesa (≥ 26.1, so rusticl on
  gfx1013 works). Community clock/fan daemon running and stable under
  sustained load.
- **Docker** and **k3s** (single-node) on the box.
- Optional, for remote access: **Tailscale**.

## Tutorial

Full detail in [TUTORIAL.md](TUTORIAL.md). Stages:

| Stage | Result |
| ----- | ------ |
| 1 | `clinfo` sees the GPU through rusticl |
| 2 | PyTorch runs on the GPU, provably faster than CPU |
| 3 | JupyterHub on k3s: CPU notebook pods with per-user persistent disks and idle culling, reachable on the LAN and over Tailscale |
| 4 | Those per-user pods use the GPU, as unprivileged pods |

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
| [BLOG.md](BLOG.md) | Short write-up: objectives, cornerstones, benchmark results, verdict. Start here. |
| [TUTORIAL.md](TUTORIAL.md) | End-to-end walkthrough. Stages 1-4. |
| [BENCHMARK.md](BENCHMARK.md) | Workload rationale, protocol, results vs cloud GPUs, and how to run it anywhere. |
| `benchmark.py`, `compare.py` | Benchmark harness (ResNet-9 / CIFAR-10). |
| `k8s/env.example`, `k8s/render.sh` | Host-specific values + the renderer that turns `k8s/**/*.yaml.tmpl` into manifests you can apply. Run `./k8s/render.sh` before the first apply. |
| `k8s/device-plugin/` | `generic-device-plugin` DaemonSet — advertises `devic.es/dri`. Required for any GPU pod (Stage 4). |
| `k8s/jupyterhub/` | The whole stack (namespace `jupyterhub`): RBAC, ConfigMap, PVC, Deployment (hub + Tailscale sidecar), Service, Ingress. |
| `k8s/jupyterhub/image/` | The Hub image: JupyterHub + `kubespawner` + `idle-culler` + `nativeauthenticator`. |
| `k8s/jupyterhub/notebook-image/` | The `bc250-notebook` image the Hub spawns per user: Fedora + rusticl + torch + patched `pytorch_ocl`. |
| `pytorch-dlprim-fix/` | The `CUSTOM_REDUCE=1` rebuild of `pytorch_ocl` that makes training ops (softmax, cross-entropy, bias gradients) compile under Fedora's stock rusticl. Patch + build script + a prebuilt `.so`. |

## License

[MIT](LICENSE) — Copyright (c) 2026 mxreyer.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
