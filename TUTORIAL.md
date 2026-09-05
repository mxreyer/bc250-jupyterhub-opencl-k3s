# Tutorial: JupyterHub with GPU-accelerated PyTorch on BC-250

This walks through turning an AsRock BC-250 mining board into a machine that
serves GPU-accelerated Jupyter notebooks from a single-node Kubernetes
cluster, reachable on the LAN over HTTP and across a Tailscale tailnet over
HTTPS.

The main objective is **learning** Kubernetes by operating something
real — everything is in plain, readable manifests, not a Helm chart.

The finished layout lives in this repo under `k8s/`. This document is the
narrative - open the manifests for reference.

---

## OpenCL, not ROCm

The BC-250's GPU reports as **gfx1013**, which is not supported by ROCm.
Instead, this build uses the **OpenCL** path: specifically, **Mesa
`rusticl`**. The out-of-tree PyTorch backend **`pytorch_ocl`** (a.k.a.
`pytorch_dlprim`) exposes OpenCL devices to PyTorch as `ocl:0`.

This also simplifies exposing the GPU to Kubernetes pods, which becomes
a plain device mount: the OpenCL path needs only the DRI render node
(`/dev/dri/renderD128`), not ROCm's `/dev/kfd`.

---

## Table of Contents

| Stage | Result |
| ----- | ------ |
| 1 | `clinfo` sees the GPU through rusticl |
| 2 | PyTorch runs on the GPU, provably faster than CPU |
| 3 | JupyterHub on k3s: CPU notebook pods with per-user persistent disks and idle culling, reachable on the LAN and over Tailscale |
| 4 | Those per-user pods use the GPU, as unprivileged pods |

---

## Stage 1 — OpenCL on bare metal

**Goal:** the GPU is visible to OpenCL through rusticl.

```bash
sudo dnf install -y mesa-libOpenCL clinfo

# rusticl only exposes the radeonsi gallium driver when explicitly told to.
# Without this the BC-250 does not appear as an OpenCL device at all. Put it
# in your shell profile — every bare-metal OpenCL step below needs it (the
# Kubernetes stages set it inside the pod instead).
export RUSTICL_ENABLE=radeonsi

clinfo -l
```

**Verify:** `clinfo -l` lists

```
Platform #0: rusticl
 `-- Device #0: AMD BC-250 (radeonsi, gfx1013, ...)
```

A `Rusticl warning: Patched Mesa libclc not detected` line on every run is
expected — [Stage 2](#stage-2--pytorch-on-the-gpu) deals with what it breaks.
Without `RUSTICL_ENABLE=radeonsi`, `clinfo -l` prints only `Platform #0:
rusticl` with no device under it.

---

## Stage 2 — PyTorch on the GPU

**Goal:** move a tensor to `ocl:0`, run a model, and **prove** the GPU is
doing the work. A silent fall-back to CPU looks exactly like success, so the
proof is a benchmark.

Use a dedicated virtualenv with the CPU build of PyTorch plus `pytorch_ocl`:

```bash
python3.12 -m venv bc250-ocl-venv
bc250-ocl-venv/bin/pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
bc250-ocl-venv/bin/pip install https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl
```

The prebuilt wheel asks the OpenCL driver for a newer feature that
Fedora's stock Mesa doesn't provide, so `softmax`, `cross_entropy`, and
bias gradients fail to compile. `pytorch_dlprim` already contains a
portable fallback for this; it's just off by default. Rebuild the
extension with the fallback on (`CUSTOM_REDUCE=1`) and drop the
resulting `pt_ocl.so` into the venv. The exact patch and a build script
are in [`pytorch-dlprim-fix/`](pytorch-dlprim-fix/).

```bash
./pytorch-dlprim-fix/build.sh
```

The script is self-contained: it builds its *own* throwaway venv under
`pytorch-dlprim-fix/scratch/` (same torch + `pytorch_ocl` as above), clones
`pytorch_dlprim`, applies `custom_reduce.patch`, and compiles. Budget ~10
minutes on the BC-250 — almost all of it the C++ build. It leaves the
rebuilt extension at `pytorch-dlprim-fix/pt_ocl.so`.

Copy that over the stock extension in the venv you made above, and delete
the compiled-kernel cache (it is keyed to the old kernel source):

```bash
cp pytorch-dlprim-fix/pt_ocl.so \
   bc250-ocl-venv/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so
rm -f ~/.dlprimitives/cache.db
```

Then, verify (with `RUSTICL_ENABLE=radeonsi` in the environment, from Stage 1 —
without it `torch.rand(..., device="ocl:0")` raises `RuntimeError: Invalid
Device #0`):

```python
import time, torch, pytorch_ocl

x = torch.rand(4096, 4096, device="ocl:0")
_ = x @ x; torch.ocl.synchronize()
t = time.time()
for _ in range(10): y = x @ x
torch.ocl.synchronize()
print(f"{2*4096**3/((time.time()-t)/10)/1e9:.0f} GFLOP/s")
```

The GPU should report ~2000+ GFLOP/s versus ~300 for `x.cpu() @ x.cpu()`.

Then confirm a small classifier trains — this is what actually exercises the
rebuilt reduction kernels (conv/linear bias gradients, `log_softmax`,
`nll_loss`):

```python
import torch, pytorch_ocl
import torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
dev = "ocl:0"

net = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8 * 32 * 32, 10),
).to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.1)

x = torch.randn(64, 3, 32, 32, device=dev)
y = torch.randint(0, 10, (64,), device=dev)

for step in range(50):
    opt.zero_grad()
    loss = F.cross_entropy(net(x), y)
    loss.backward()
    opt.step()
    if step == 0:
        first = loss.item()

print(f"loss {first:.3f} -> {loss.item():.3f}")
```

- **Without** the rebuilt `pt_ocl.so`, the first `backward()` throws an
  OpenCL *compile* error — `use of undeclared identifier
  'work_group_reduce_add'`.
- **With** it, the loss falls steadily (e.g. `2.29 -> 0.01`). Every op ran on
  the GPU.

A `Rusticl warning: Patched Mesa libclc not detected` line on startup is
expected and harmless — the whole point of the rebuild is to not need that
patched libclc.

---

## Stage 3 — Kubernetes: JupyterHub on k3s (no GPU yet)

**Goal:** JupyterHub running as pods on a k3s cluster, reachable in a browser
over the LAN. A user signs up, an admin approves, and logging in spawns a
per-user notebook pod that can run `1 + 1`. No GPU yet.

First, install k3s

```bash
curl -sfL https://get.k3s.io | sh -
```

k3s is real, upstream-conformant Kubernetes in one binary. It ships CoreDNS,
the **Traefik** ingress controller, a **local-path** storage provisioner, and
`metrics-server`.

### Two images, built locally

Build both with Docker and import them into k3s's containerd (separate image
stores):

- **`k8s/jupyterhub/image/`** — the Hub: upstream JupyterHub + `kubespawner` +
  `jupyterhub-idle-culler` + `nativeauthenticator`.
- **`k8s/jupyterhub/notebook-image/`** — the per-user server: **`FROM
  fedora:44`** (matched to the host so the OpenCL userspace lines up in Stage
  4), torch 2.4.0 CPU, the patched `pytorch_ocl` from Stage 2, `jupyterlab`,
  and `jupyterhub` for the `jupyterhub-singleuser` launcher.

```bash
k8s/jupyterhub/image/build.sh
k8s/jupyterhub/notebook-image/build.sh
```

Neither image is in a registry, so the manifests set `imagePullPolicy: Never`.

### Host-specific values

The committed manifests carry `${NODE_LAN_IP}` / `${TS_*}` / `${HUB_ADMIN}`
placeholders. Once, before applying anything:

```bash
cp k8s/env.example k8s/env.local     # then edit env.local
./k8s/render.sh                       # writes the real k8s/**/<name>.yaml
```

### Namespace

```bash
sudo kubectl apply -f k8s/jupyterhub/00-namespace.yaml
```

### Secrets

The Secret is never committed; `20-secret.example.txt` documents the keys:

```bash
sudo kubectl -n jupyterhub create secret generic hub-secrets \
  --from-literal=CONFIGPROXY_AUTH_TOKEN=$(openssl rand -hex 32) \
  --from-literal=JUPYTERHUB_CRYPT_KEY=$(openssl rand -hex 32)
```

The `tailscale` sidecar in `50-deployment.yaml` also needs a `tailscale-auth`
Secret before you apply anything else. Kubernetes only adds a pod to a
Service's endpoints once **every** container in it is Ready — so until this
Secret exists, the `tailscale` container sits in `CreateContainerConfigError`,
the Hub pod never reaches `Ready`, and Traefik returns a bare `503` for
everyone, not just a degraded sidecar. Grab an auth key from [Settings →
Keys](https://login.tailscale.com/admin/settings/keys) → *Generate auth key*
(a `tskey-auth-...` string) and create the Secret now:

```bash
sudo kubectl -n jupyterhub create secret generic tailscale-auth \
  --from-literal=authkey=tskey-auth-XXXXXXXX
```

(`34-tailscale-auth.example.txt` documents the same command, for reference —
don't apply that file itself. See the Tailscale section below for what this
Secret is used for.) If you'd rather skip Tailscale for now, comment out the
`tailscale` container block in `50-deployment.yaml.tmpl` (then re-run
`./k8s/render.sh`) instead — the Hub itself doesn't need it.

### The objects (`k8s/jupyterhub/`)

| File | Kind | Purpose |
| ---- | ---- | ------- |
| `00-namespace.yaml` | Namespace | `jupyterhub` — holds the Hub *and* every spawned user pod |
| `10-rbac.yaml` | ServiceAccount + Role + RoleBinding | let the Hub drive the k8s API — **in this namespace only** |
| `20-secret.example.txt` | *(doc only)* | documents the `hub-secrets` keys — the real Secret is hand-created above, never committed |
| `30-configmap.yaml` | ConfigMap | the entire `jupyterhub_config.py` |
| `34-tailscale-auth.example.txt` | *(doc only)* | documents the `tailscale-auth` Secret command — the real Secret is hand-created above, never committed |
| `35-tailscale-serve.yaml` | ConfigMap | `serve.json` for the Tailscale sidecar — tailnet `:443` → `127.0.0.1:8000` |
| `40-pvc.yaml` | PVC | the Hub's SQLite state — user table *and* NativeAuthenticator credentials |
| `50-deployment.yaml` | Deployment | one Hub pod: JupyterHub + its HTTP proxy + the culler + the Tailscale sidecar |
| `60-service.yaml` | Service | stable in-cluster address — `:8000` public, `:8081` Hub API |
| `70-ingress.yaml` | Ingress | tells Traefik: `hub.<LAN-IP>.sslip.io` → the Service |

**Core concepts:**

- A **Deployment** manages a **ReplicaSet** manages **Pods**, matched by
  label. The Hub is a single-replica Deployment with `strategy: Recreate`
  because it holds a `ReadWriteOnce` state PVC — never two Hub pods at once.
- A pod's IP is disposable. A **Service** gives a fixed virtual IP and DNS
  name (`hub.jupyterhub.svc.cluster.local`) that tracks whichever pods match.
- An **Ingress** routes for Traefik: match the HTTP `Host` header, forward to
  a Service. The wildcard host `hub.<LAN-IP>.sslip.io` needs no DNS setup —
  `sslip.io` resolves `anything.192.168.1.100.sslip.io` straight to that IP.
- **Persistent Hub state.** A **PersistentVolumeClaim** requests durable disk;
  k3s's `local-path` provisioner satisfies it with a directory on the node.
  `hub-state` holds the SQLite DB (user table + bcrypt credentials) and the
  cookie secret — lose it and every user's tokens are orphaned.
- **RBAC.** A pod can't touch the Kubernetes API by default. The Hub's
  ServiceAccount is bound to a **Role** (namespaced) granting exactly the
  verbs KubeSpawner needs: create/delete pods, PVCs and Secrets; read events
  and pod logs. A `Role`, not `ClusterRole`, so the powers stop at the
  `jupyterhub` namespace — check with `sudo kubectl auth can-i
  --as=system:serviceaccount:jupyterhub:hub create pods -n jupyterhub`.
- **`jupyterhub_config.py`** is JupyterHub's main config file (auth, spawner,
  storage, idle-culler — everything below). It isn't a standalone file in this
  repo: it's inlined as the `jupyterhub_config.py` key inside the `hub-config`
  ConfigMap (`30-configmap.yaml`), mounted read-only into the Hub pod at
  `/srv/jupyterhub/config/`. The container's command loads it explicitly —
  `jupyterhub --config /srv/jupyterhub/config/jupyterhub_config.py`. So to
  change it, edit `30-configmap.yaml.tmpl`, re-run `./k8s/render.sh`,
  re-`apply`, then bump `config-revision` in `50-deployment.yaml.tmpl`
  (or `kubectl rollout restart`) — ConfigMap edits don't restart the pod on
  their own.
- **The spawner.** `KubeSpawner` turns "user logs in" into a Kubernetes API
  call that creates `jupyter-<user>` and `claim-<user>`. The per-user pod spec
  lives in `jupyterhub_config.py` as spawner settings, not as a static
  Deployment.
- **Authentication.** `NativeAuthenticator`: users self-register at
  `/hub/signup`, an admin approves at `/hub/authorize`, everyone changes their
  own password at `/hub/change-password`. Credentials (bcrypt) sit in the Hub
  DB on the PVC. One non-obvious setting: `c.Authenticator.allow_all = True`
  is required — JupyterHub 5's outer `check_allowed()` denies everyone
  otherwise, and NativeAuthenticator's signup/approve flow is the real gate.
- **Per-user persistent storage.** `c.KubeSpawner.storage_pvc_ensure = True`
  makes KubeSpawner create a 10Gi `claim-<user>` PVC per login and mount it at
  `/home/jovyan` — k3s's `local-path` provisioner backs it with a directory on
  the node's disk. Notebooks, `pip install`s, and the OpenCL kernel cache
  survive a server restart. `ReadWriteOnce` (one node at a time) is moot on
  this single-node box, but it's why the Hub and every per-user pod are
  "recreate, don't roll" — two pods can't mount the same claim at once.
- **Idle culling.** `jupyterhub-idle-culler` (registered in
  `c.JupyterHub.services` with a scoped role in `c.JupyterHub.load_roles`)
  runs inside the Hub pod and reaches it over loopback, stopping any server
  idle longer than `--timeout=3600` (one hour) — freeing the `devic.es/dri`
  slot and ~1GiB of RAM an idle notebook would otherwise hold.
- **Remote access over Tailscale.** The `tailscale` sidecar shares the Hub
  pod's network namespace and runs Tailscale Serve, which terminates TLS with
  an auto-provisioned certificate for `<node>.<tailnet>.ts.net` and proxies to
  `127.0.0.1:8000` (the Hub's proxy) per `serve.json`
  (ConfigMap `35-tailscale-serve.yaml`); `AllowFunnel: false` keeps it
  tailnet-only. Key settings: `TS_USERSPACE=true` (userspace WireGuard — no
  `/dev/net/tun`, no extra capabilities, same as Tailscale's own Kubernetes
  operator); `TS_KUBE_SECRET=""` (otherwise the image detects Kubernetes and
  crash-loops trying to store node state in a Secret it has no RBAC for);
  `TS_STATE_DIR` on a subdirectory of the Hub's own PVC, so a pod restart
  reconnects as the **same** tailnet node; and
  `c.JupyterHub.public_url = "https://…ts.net/"` in the ConfigMap — without it
  the Hub sees plain HTTP from the sidecar, emits `http://` redirects, and the
  login loops.

### Apply and verify

Comment out the GPU block in `30-configmap.yaml.tmpl` (flagged in the file)
and re-run `./k8s/render.sh`.
There is no device plugin yet, so with it left in, every spawned user pod
requests `devic.es/dri` and stays `Pending` forever. Stage 4 turns it back on.

Then apply the rest:

```bash
sudo kubectl apply -f k8s/jupyterhub/
```

Verify:

```bash
sudo kubectl -n jupyterhub get deploy,rs,pod,svc,ingress,pvc
curl -s -o /dev/null -w '%{http_code}\n' http://hub.<LAN-IP>.sslip.io/hub/login   # 200 from the Hub
```

### Log in

Open `http://hub.<LAN-IP>.sslip.io/hub/login` and sign up as
`${HUB_ADMIN}` — NativeAuthenticator auto-authorizes any username listed
in `c.Authenticator.admin_users`, so that one account needs no separate
approval. Log in and watch

```bash
sudo kubectl -n jupyterhub get pod -w
```

create `jupyter-<you>`; in it, run `1 + 1`. If the kernel responds, the
path browser → Traefik → Hub → proxy → singleuser pod works.

> On Fedora, `firewall-cmd --reload` flushes and reinstalls firewalld's
> nftables/iptables rules, wiping whatever k3s's CNI (flannel) had already set
> up — pods can stop reaching each other until k3s re-adds its rules. Add k3s's
> pod/service CIDRs to firewalld's `trusted` zone (no filtering for that purely
> internal traffic) and restart k3s to force it to reinstall its rules:
>
> ```bash
> sudo firewall-cmd --permanent --zone=trusted --add-source=10.42.0.0/16  # pod CIDR
> sudo firewall-cmd --permanent --zone=trusted --add-source=10.43.0.0/16  # service CIDR
> sudo firewall-cmd --reload
> sudo systemctl restart k3s
> ```
>
> (`10.42.0.0/16`/`10.43.0.0/16` are k3s's defaults — use your own if you
> customized `--cluster-cidr`/`--service-cidr` at install time. Verify with
> `sudo firewall-cmd --zone=trusted --list-sources`.)

Confirm per-user persistence and idle culling:

```bash
sudo kubectl -n jupyterhub get pvc -w        # claim-<you> appears on first login
```

Write a file in `/home/jovyan`, stop your server from the Hub UI and start it
again — the file is still there. Leave a server idle past the timeout and
watch the culler delete `jupyter-<you>` (`sudo kubectl -n jupyterhub get pod
-w`).

### Confirm Tailscale access

Skip if you commented out the sidecar.

```bash
sudo kubectl -n jupyterhub exec deploy/hub -c tailscale -- tailscale serve status
curl -s -o /dev/null -w '%{http_code}\n' https://<node>.<tailnet>.ts.net/hub/login
```

`serve status` shows `https://…ts.net → http://127.0.0.1:8000`; the curl
returns `200` with a valid certificate. Log in from a tailnet device and run
a cell.


---

## Stage 4 — The GPU inside the per-user pods

**Goal:** the pods JupyterHub spawns run PyTorch on the BC-250 GPU, and do so
as ordinary unprivileged pods.

### Why the notebook image is `FROM fedora:44`

The image you built in Stage 3 (`k8s/jupyterhub/notebook-image/`) is already
GPU-ready: it's built from the **same distro and version as the host** so its
Mesa userspace (`mesa-libOpenCL`, `mesa-dri-drivers`, `libclc`, `ocl-icd`)
lines up with the host kernel's `amdgpu`, which is what rusticl talks to
through `/dev/dri/renderD128`. Nothing to rebuild here — Stage 4 only wires
the device to it.

### Handing over the device (`k8s/device-plugin/`)

A plain `hostPath` mount of `/dev/dri` makes the device *file* visible but the
kernel's **device cgroup** still blocks I/O on it. The right tool is a
**device plugin**: a small DaemonSet that advertises the device to the kubelet
as a schedulable resource. Pods *request* it and the kubelet wires it in with
exactly the right permission and nothing more.

This build uses **`generic-device-plugin`** (config-driven, no code). It
advertises `/dev/dri/renderD128` as `devic.es/dri`. Only the plugin DaemonSet
runs privileged; workload pods stay unprivileged.

```bash
sudo kubectl apply -f k8s/device-plugin/
sudo kubectl describe node | grep 'devic.es/dri'
```

Expect two lines reading `10` — from the node's `Capacity:` and
`Allocatable:` sections. `10`, not `1`: the plugin is configured to let up to
10 pods share the single render node (`count: 10` in
`generic-device-plugin.yaml`) — rusticl contexts coexist fine on one device.

If any per-user pod is already running with the GPU wired up, the same grep
also matches a third line from the node's `Allocated resources:` table —
formatted `devic.es/dri  <requests>  <limits>`, e.g. `1  1` for one active
pod. That line reflects current usage, not the `10` total, so don't read it
as a mismatch.

### What the spawner config adds

Uncomment the GPU block in `k8s/jupyterhub/30-configmap.yaml.tmpl` that you
disabled in Stage 3 (flagged in the file), re-run `./k8s/render.sh`, re-apply
it, and `sudo kubectl -n jupyterhub rollout restart deploy/hub`. The block:

```python
c.KubeSpawner.extra_resource_limits     = {"devic.es/dri": "1"}   # request the render node
c.KubeSpawner.extra_resource_guarantees = {"devic.es/dri": "1"}
c.KubeSpawner.environment      = {"RUSTICL_ENABLE": "radeonsi"}
c.KubeSpawner.supplemental_gids = [105]                           # host "render" group
c.KubeSpawner.mem_limit = "6G"                                    # GPU buffers count as pod memory
```

Requesting the resource also means a user pod won't schedule at all if the
plugin isn't running — no silent CPU-only fallback.

### Verify

Start a fresh server from the Hub, then run the GPU check inside it:

```bash
sudo kubectl -n jupyterhub cp k8s/jupyterhub/gpu-check.py jupyter-<you>:/tmp/gpu-check.py
sudo kubectl -n jupyterhub exec jupyter-<you> -- python /tmp/gpu-check.py
```

GPU GFLOP/s should be several times the CPU figure — the same numbers as bare
metal. Confirm the pod's `securityContext` has no `privileged`, and `/dev/dri/`
inside the pod contains only `renderD128`.

---

## Done!

That's the whole system: see [README.md](README.md#architecture-end-to-end)
for the end-to-end architecture diagram.

---

## Managing users

JupyterHub authenticates with **NativeAuthenticator** (see the Authentication
bullet in Stage 3's Core concepts) — everyone manages their own account, so
day-to-day administration rarely touches `kubectl`:

- **`/hub/signup`** — anyone creates an account (username + password).
  `c.NativeAuthenticator.open_signup = False`, so new accounts sit pending
  until an admin approves them.
- **`/hub/authorize`** — an admin approves (or denies) pending signups.
- **`/hub/change-password`** — any logged-in user rotates their own password.
- **`/hub/admin`** — admin dashboard: list/approve/delete users, start/stop
  servers.
- **Admins** are the usernames in `c.Authenticator.admin_users`
  (`${HUB_ADMIN}`, set in `k8s/env.local`) — auto-approved on signup, and the
  only accounts that can reach `/hub/authorize` and `/hub/admin`. Add more by
  editing that set in `30-configmap.yaml.tmpl`, re-running `render.sh`,
  re-`apply`-ing, and bumping `config-revision` in
  `50-deployment.yaml.tmpl`.
- Password rules and brute-force lockout live in the same ConfigMap:
  minimum length 10, a common-password check, and a 10-minute lockout after 5
  failed attempts (`c.NativeAuthenticator.allowed_failed_logins` /
  `seconds_before_next_try`).
- Credentials (bcrypt) and approval state sit in the Hub's SQLite DB on the
  `hub-state` PVC — durable across pod restarts, lost only if that PVC is
  deleted.

---

## Operating it

```bash
# everything at a glance
sudo kubectl -n jupyterhub get pods,pvc

# front doors
#   LAN:     http://hub.<LAN-IP>.sslip.io
#   tailnet: https://<node>.<tailnet>.ts.net

# user management (signup/approve/admin) — see "Managing users" above

# rebuild the hub / notebook image after changing it, then restart the Hub
# (the new notebook image is picked up on the next server start)
k8s/jupyterhub/image/build.sh
k8s/jupyterhub/notebook-image/build.sh
sudo kubectl -n jupyterhub rollout restart deploy/hub

# re-apply all manifests (run render.sh first if env.local or a *.tmpl changed;
# the *.example.txt templates are skipped by extension, so real Secrets are
# never clobbered)
./k8s/render.sh
sudo kubectl apply -f k8s/device-plugin/ -f k8s/jupyterhub/

# GPU sanity from inside a running user pod
sudo kubectl -n jupyterhub exec jupyter-<user> -- python /tmp/gpu-check.py

# after any firewalld reload on this box
sudo systemctl restart k3s
```

- Comparing the board against cloud GPUs: [`BENCHMARK.md`](BENCHMARK.md)
  (`benchmark.py` runs the same ResNet-9/CIFAR-10 workload on `ocl:0`, `cuda`,
  or `cpu`)

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
