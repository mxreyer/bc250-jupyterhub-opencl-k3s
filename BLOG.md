# Turning a mining board into a multi-user ML box

The AsRock BC-250 is a crypto-mining board built around a cut-down
variant of the PlayStation 5's APU: RDNA2, 40 compute units (after
[unlock](https://github.com/duggasco/bc250-40cu-unlock)), 16 GB of
unified memory. The boards were dumped on the surplus market, and they
are now some of the cheapest 16 GB compute you can put on a desk.

They are also hardware that AMD's own ML stack refuses to touch. The GPU
reports as **gfx1013**, which ROCm does not support.

I wanted to know how far you could push one anyway. The answer turned into
a small ML server: JupyterHub on Kubernetes, GPU-accelerated PyTorch,
multiple users, remote access over HTTPS.

## Objectives

1. **Get PyTorch onto the GPU** without ROCm.
2. **Serve it to more than one person** — real accounts, persistent
   storage, notebooks in a browser.
3. **Learn Kubernetes properly** by operating something real, in plain
   manifests rather than a Helm chart someone else wrote.
4. **Measure it honestly** against cloud GPUs, rather than hand-waving
   about "good enough".

## The four cornerstones

### OpenCL instead of ROCm

ROCm is a dead end on gfx1013, so the GPU is reached through
**rusticl**, Mesa's OpenCL implementation, with the out-of-tree
[**`pytorch_ocl`**
backend](https://github.com/artyom-beilis/pytorch_dlprim) exposing it to
PyTorch as `ocl:0`. This turned out to be a strategic win beyond just
working: the OpenCL path needs only the DRI render node
(`/dev/dri/renderD128`), not ROCm's `/dev/kfd`, which makes handing the
GPU to a container dramatically simpler.

### One patch to make training work

Out of the box, inference ran and training didn't. Every kernel that
reduces values across a work-group — softmax, cross-entropy, bias
gradients — failed to *compile*.

The cause is a supply-chain mismatch, not a bug in anyone's code. Those
kernels call the OpenCL 2.0 `work_group_reduce_add()` built-ins, and
rusticl only declares them when it detects Mesa's own patched fork of
libclc. Fedora ships plain upstream libclc, so the built-ins vanish.

`pytorch_ocl`'s own dependency already had a portable fallback — a
hand-written reduction using local memory and barriers, needing nothing
past OpenCL 1.2 — sitting behind a `CUSTOM_REDUCE` flag that defaults to
off. The entire fix is flipping that flag and rebuilding one Python
extension. Nothing on the host is touched.

### The GPU inside an unprivileged pod

The lazy way to give a container a GPU is a `hostPath` mount and
`privileged: true`. That makes the device file visible while the kernel's
device cgroup still blocks I/O on it, and hands the notebook the keys to
every device on the box.

The right tool is a **device plugin** — a small DaemonSet that advertises
the render node to the kubelet as a schedulable resource. Pods *request*
`devic.es/dri` and the kubelet wires it in with exactly the right
permission and nothing more. Only the plugin runs privileged; every
notebook pod stays unprivileged. Requesting a real resource also means a
user pod won't schedule at all if the GPU is unavailable — no silent
fallback to CPU that you discover three hours into a training run.

### JupyterHub that other people can actually use

k3s (single-node upstream Kubernetes) runs a JupyterHub whose spawner
turns "user logs in" into a Kubernetes API call: a per-user pod plus a
per-user 10 GB persistent volume, created on demand and culled after an
hour idle so the GPU is freed. Users self-register and an admin approves
them. A Tailscale sidecar inside the Hub pod terminates real HTTPS for
remote access, in userspace, with no extra privileges.

## Benchmark

Identical workload everywhere — **ResNet-9 on CIFAR-10**, 16 epochs, batch
128, FP32. The BC-250 was measured at both 24 and 40 compute units; cloud
cards get a second run with mixed precision (`--amp`) for their realistic
best case, which the BC-250 cannot do at all.

**Training throughput**

| Platform | Precision | img/s | vs BC-250 |
| --- | --- | ---: | ---: |
| BC-250, CPU only | fp32 | 70 | 0.09× |
| BC-250, 24 CU | fp32 | 532 | 0.68× |
| **BC-250, 40 CU** | **fp32** | **786** | **1.0×** |
| T4 (Colab) | fp32 | 2,294 | 2.9× |
| T4 (Colab) | amp | 4,441 | 5.6× |
| A100 (Colab) | fp32 | 7,377 | 9.4× |
| A100 (Colab) | amp | 18,690 | 24× |

**Inference throughput** (batch 128)

| Platform | Precision | img/s | vs BC-250 |
| --- | --- | ---: | ---: |
| BC-250, 40 CU | fp32 | 3,892 | 1.0× |
| T4 (Colab) | fp32 | 7,217 | 1.9× |
| T4 (Colab) | amp | 7,355 | 1.9× |
| A100 (Colab) | amp | 49,612 | 13× |

Three things stand out.

**Accuracy lands in the same place everywhere** — 0.922 to 0.927 across
every platform. The OpenCL stack is numerically correct, which is the
result that matters most.

**The gap is software, not silicon.** At 40 CUs and ~1.5 GHz the BC-250 is
theoretically a ~6–8 TFLOP/s FP32 part — genuinely T4-class hardware. It
delivers roughly a third of a T4's training throughput. That difference is
kernel quality and maturity.

**Inference is the sweet spot.** At 1.9× behind a T4, the board is
perfectly respectable for serving. Training is where immature kernels and
the total absence of mixed precision hurt most.

## Verdict

For surplus mining hardware running on a driver stack nobody intended for
compute, roughly one-third of a cloud T4 in training and half in inference
is a good outcome — and it came down to one build flag.

It is not a cloud GPU replacement. There is no mixed precision, no tensor
cores, and a maturing kernel library. If you need to train fast, rent an
A100 — it is 24× quicker and you will spend less than the electricity.

What it *is* very good at is being **always-on and yours**. No session
limits, no idle timeouts disconnecting you, no per-hour meter, no upload
step. For fine-tuning small models, running inference, teaching, and
learning Kubernetes on something with real consequences, a permanently
available 16 GB GPU under your desk beats a faster one you have to keep
re-renting.

---

Full walkthrough, manifests, and the benchmark harness:
[TUTORIAL.md](TUTORIAL.md) · [BENCHMARK.md](BENCHMARK.md) ·
[README.md](README.md)

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
