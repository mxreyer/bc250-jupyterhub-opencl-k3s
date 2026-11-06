# Setup: JupyterHub with GPU-accelerated PyTorch on BC-250

This walks through turning an AsRock BC-250 mining board into a machine that
serves GPU-accelerated Jupyter notebooks from a single-node Kubernetes
cluster, reachable on the LAN over HTTP and across a Tailscale tailnet over
HTTPS.

The main objective is **learning** Kubernetes by operating something
real — everything is in plain, readable manifests, not a Helm chart.

The finished layout lives in this repo under `k8s/`. This document is the
narrative — open the manifests for reference.

## Before you start: what the host needs

The notebook image built in step 2 carries the whole GPU software stack
(Mesa rusticl, Mesa's patched libclc, torch 2.4.0, the patched
`pytorch_ocl`, `RUSTICL_ENABLE=radeonsi`). The only thing a pod takes from
the host is the `amdgpu` kernel driver behind `/dev/dri/renderD128`. So the
host needs:

- **AsRock BC-250 on Fedora** with the `amdgpu` driver loaded:
  `ls /dev/dri/` shows `renderD128`.
- The community **clock/fan daemon** running and stable under sustained
  load. Training in a pod loads the GPU exactly as it would on bare metal.
- A **`render` group with GID 105**: `getent group render` prints
  `render:x:105:`. The spawner adds pods to that GID to open the render
  node. If yours differs, change `supplemental_gids` in
  `k8s/jupyterhub/30-configmap.yaml.tmpl` and `--group-add` in the Docker
  smoke test (step 3).
- Optional: the 40 CU / 2.0 GHz unlock from
  [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock).
  Everything works at the stock 24 CUs, just slower.

Mesa, OpenCL, PyTorch and `pt_ocl.so` do **not** need to be installed on the
host. The GPU check that counts is the Docker smoke test in step 3, run against
the freshly built notebook image: it exercises the same device, driver and
userspace a pod will use.

**Recommended, not required:** working through Stages 1-2 of
[**pytorch-gpu-bc250**](https://github.com/mxreyer/pytorch-gpu-bc250/blob/main/SETUP.md)
on the host first. A bare-metal setup is the easiest place to debug the GPU,
and its errors are far more readable than a failing pod's.

Why OpenCL and not ROCm matters here: the BC-250's GPU (**gfx1013**) is not
a ROCm target, so PyTorch reaches it through Mesa's **rusticl** and the
`pytorch_ocl` backend. The OpenCL path needs only the DRI render node
(`/dev/dri/renderD128`), not ROCm's `/dev/kfd`, which turns exposing the GPU
to Kubernetes pods into a plain device request.

## Overview

**Goal:** JupyterHub running as pods on a k3s cluster, reachable over the
LAN and over Tailscale. A user signs up, an admin approves, and logging in
spawns a per-user notebook pod, with its own persistent disk, that runs
PyTorch on the BC-250 GPU as an ordinary unprivileged pod. Idle servers are
culled after an hour.

The steps build the system bottom-up, and each ends with a check, so a
failure always points at the layer just added:

| Step | Result |
| ---- | ------ |
| 1 | k3s installed |
| 2 | Hub and notebook images built and imported into k3s |
| 3 | The notebook image reaches the GPU under plain Docker |
| 4 | Kubernetes advertises the GPU as a schedulable resource |
| 5 | *(reading)* what is in `k8s/jupyterhub/` and why |
| 6 | JupyterHub deployed and answering on the LAN |
| 7 | Logged in: a per-user pod with a persistent disk and idle culling |
| 8 | That pod computes on the GPU |
| 9 | Remote access over Tailscale |

## 1. Install k3s

```bash
curl -sfL https://get.k3s.io | sh -
```

k3s is real, upstream-conformant Kubernetes in one binary. It ships CoreDNS,
the **Traefik** ingress controller, a **local-path** storage provisioner, and
`metrics-server`.

## 2. Build the two images

Build both with Docker and import them into k3s's containerd (separate image
stores):

- **`k8s/jupyterhub/image/`** — the Hub: upstream JupyterHub + `kubespawner` +
  `jupyterhub-idle-culler` + `nativeauthenticator`.
- **`k8s/jupyterhub/notebook-image/`** — the per-user server: **`FROM
  fedora:45`** (Mesa 26.2 with rusticl, plus Mesa's patched libclc dropped in
  over Fedora's — see that directory's README), torch 2.4.0 CPU, the patched
  `pytorch_ocl` from
  [pytorch-gpu-bc250](https://github.com/mxreyer/pytorch-gpu-bc250/blob/main/SETUP.md), `jupyterlab`, and `jupyterhub` for the
  `jupyterhub-singleuser` launcher.

```bash
k8s/jupyterhub/image/build.sh
k8s/jupyterhub/notebook-image/build.sh
```

Neither image is in a registry, so the manifests set `imagePullPolicy: Never`.

### Why the notebook image is `FROM fedora:45`

The notebook image (`k8s/jupyterhub/notebook-image/`) is GPU-ready on its
own: it carries its own Mesa userspace (`mesa-libOpenCL`,
`mesa-dri-drivers`, `libclc`, `ocl-icd`), and the only piece it shares with
the host is the `amdgpu` kernel driver behind `/dev/dri/renderD128`. So the
container's Fedora does not have to match the host's — Fedora 45's Mesa 26.2
runs fine on a Fedora 44 kernel, and is what the benchmark numbers were
measured on. All Kubernetes has to do is hand the device to the pod.

## 3. Check the GPU from the notebook image

Before any Kubernetes is involved, confirm the image reaches the GPU
with plain Docker:

```bash
sudo docker run --rm -it \
  --device /dev/dri/renderD128 --group-add 105 \
  bc250-notebook:latest \
  bash -lc 'clinfo -l && python -c "import torch,pytorch_ocl; \
    print(torch.randn(1024,1024,device=\"ocl:0\").sum())"'
```

Expect `clinfo -l` to list `Platform #0: rusticl` with the BC-250 device and
no `Rusticl warning` line, then a number with no compile error. If this fails,
fix it here (host driver, `render` GID) before going further; inside a pod
the same failure is much harder to read.

## 4. Hand the GPU to Kubernetes (`k8s/device-plugin/`)

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

Checking this now, before any user pod exists, means a GPU failure in step 8
is about the pod itself, not a node with no GPU to give.

Later, once per-user pods are running, the same grep also matches a third
line from the node's `Allocated resources:` table — formatted
`devic.es/dri  <requests>  <limits>`, e.g. `1  1` for one active pod. That
line reflects current usage, not the `10` total, so don't read it as a
mismatch.

## 5. The objects (`k8s/jupyterhub/`)

| File | Kind | Purpose |
| ---- | ---- | ------- |
| `00-namespace.yaml` | Namespace | `jupyterhub` — holds the Hub *and* every spawned user pod |
| `10-rbac.yaml` | ServiceAccount + Role + RoleBinding | let the Hub drive the k8s API — **in this namespace only** |
| `20-secret.example.txt` | *(doc only)* | documents the `hub-secrets` keys — the real Secret is hand-created below, never committed |
| `30-configmap.yaml` | ConfigMap | the entire `jupyterhub_config.py` |
| `34-tailscale-auth.example.txt` | *(doc only)* | documents the `tailscale-auth` Secret command — the real Secret is hand-created below, never committed |
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
- **Ingress.** The Service's IP is cluster-internal, so a LAN browser can't
  reach it. An **Ingress** is a routing rule, and the **ingress controller**
  enforces it. That controller is Traefik, which k3s bundles, listening on the
  node's `:80`/`:443`. The rule in `70-ingress.yaml` reads "requests whose
  HTTP `Host` header is `hub.<LAN-IP>.sslip.io`, any path → Service `hub`,
  port `http` (8000)". So the path is browser → `<LAN-IP>:80` (Traefik) →
  Service → Hub pod. Traefik matches on the **hostname**, not the IP:
  `http://<LAN-IP>/` sends the wrong `Host` and gets a 404. Only `:8000` is
  routed; the Hub API (`:8081`) stays in-cluster. The rule has no `tls:`
  block, so the LAN path is plain HTTP; HTTPS comes from the Tailscale
  sidecar, which bypasses the Ingress.
- **sslip.io** is an ordinary domain under `.io`. Its DNS
  server answers with whatever IP is embedded in the name you look up:
  `hub.192.168.1.100.sslip.io` (or `hub.192-168-1-100.sslip.io`) → `192.168.1.100`.
  That gives the Ingress a real hostname to match without local DNS,
  `/etc/hosts` edits or a bought domain. `render.sh` fills in `NODE_LAN_IP`
  from `k8s/env.local`. Caveats:
  - The lookup goes to sslip.io's public DNS, so clients need internet access.
    Some routers' DNS-rebinding protection drops public names that resolve to
    private IPs.
  - The IP is part of the name. If the box's LAN IP changes, re-render and
    re-apply; a DHCP reservation for the box avoids this.
  - No practical TLS. A Let's Encrypt cert would need the box reachable from
    the internet, which is another reason HTTPS goes through Tailscale.
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
- **GPU wiring.** The spawner settings in `jupyterhub_config.py` that put a
  user pod on the GPU:

  ```python
  c.KubeSpawner.extra_resource_limits     = {"devic.es/dri": "1"}   # request the render node
  c.KubeSpawner.extra_resource_guarantees = {"devic.es/dri": "1"}
  c.KubeSpawner.environment      = {"RUSTICL_ENABLE": "radeonsi"}
  c.KubeSpawner.supplemental_gids = [105]                           # host "render" group
  c.KubeSpawner.mem_limit = "6G"                                    # GPU buffers count as pod memory
  ```

  Requesting `devic.es/dri` makes the kubelet wire in the render node from
  step 4. It also means a user pod won't schedule at all if the plugin isn't
  running — it stays `Pending`, with no silent CPU-only fallback.
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

## 6. Deploy JupyterHub

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
don't apply that file itself. See *Remote access over Tailscale* in step 5's
Core concepts for what this Secret is used for.) If you'd rather skip
Tailscale for now, comment out the `tailscale` container block in
`50-deployment.yaml.tmpl` (then re-run
`./k8s/render.sh`) instead — the Hub itself doesn't need it.

### Apply and verify

```bash
sudo kubectl apply -f k8s/jupyterhub/
```

Verify:

```bash
sudo kubectl -n jupyterhub get deploy,rs,pod,svc,ingress,pvc
curl -s -o /dev/null -w '%{http_code}\n' http://hub.<LAN-IP>.sslip.io/hub/login   # 200 from the Hub
```

## 7. Log in

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

## 8. Check the GPU from a user pod

Start a server from the Hub (the culler may have stopped the one from step 7),
then run the GPU check inside it:

```bash
sudo kubectl -n jupyterhub cp k8s/jupyterhub/gpu-check.py jupyter-<you>:/tmp/gpu-check.py
sudo kubectl -n jupyterhub exec jupyter-<you> -- python /tmp/gpu-check.py
```

GPU GFLOP/s should be several times the CPU figure — the same numbers
pytorch-gpu-bc250 reports on bare metal. Confirm the pod's `securityContext`
has no `privileged`, and `/dev/dri/` inside the pod contains only
`renderD128`.

## 9. Confirm Tailscale access

Skip if you commented out the sidecar.

```bash
sudo kubectl -n jupyterhub exec deploy/hub -c tailscale -- tailscale serve status
curl -s -o /dev/null -w '%{http_code}\n' https://<node>.<tailnet>.ts.net/hub/login
```

`serve status` shows `https://…ts.net → http://127.0.0.1:8000`; the curl
returns `200` with a valid certificate. Log in from a tailnet device and run
a cell.

## Done!

That's the whole system: see [README.md](README.md#architecture-end-to-end)
for the end-to-end architecture diagram.

## Managing users

JupyterHub authenticates with **NativeAuthenticator** (see the Authentication
bullet in step 5's Core concepts) — everyone manages their own account, so
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

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
