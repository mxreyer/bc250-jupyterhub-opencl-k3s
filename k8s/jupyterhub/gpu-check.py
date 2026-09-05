"""Stage 4 GPU check - run inside a spawned user pod.

Proves the pod is really using the BC-250 GPU through OpenCL, not silently
falling back to CPU, by comparing matmul throughput. A silent CPU fallback
would show near-identical numbers; a working GPU is several times faster and
well above what this CPU can physically reach.

    sudo kubectl -n jupyterhub cp k8s/jupyterhub/gpu-check.py jupyter-<user>:/tmp/gpu-check.py
    sudo kubectl -n jupyterhub exec jupyter-<user> -- python /tmp/gpu-check.py
"""
import time
import torch
import pytorch_ocl  # noqa: F401  (registers the "ocl" device)

DEV = "ocl:0"


def bench(x, iters):
    # one warm-up + sync so we time steady-state, not first-call compilation
    _ = x @ x
    sync()
    t0 = time.time()
    for _ in range(iters):
        y = x @ x
    sync()
    dt = (time.time() - t0) / iters
    gflops = 2 * x.shape[0] ** 3 / dt / 1e9
    return dt, gflops, float(y.sum())


def sync():
    if hasattr(torch, "ocl"):
        torch.ocl.synchronize()


for n in (1024, 2048, 4096):
    xg = torch.rand(n, n, device=DEV)
    dt_g, gf_g, checksum = bench(xg, 20)
    xc = xg.cpu()
    dt_c, gf_c, _ = bench(xc, 3)
    print(
        f"{n:>5}^3  "
        f"GPU {dt_g*1e3:7.1f} ms {gf_g:6.0f} GFLOP/s   "
        f"CPU {dt_c*1e3:7.1f} ms {gf_c:6.0f} GFLOP/s   "
        f"speedup x{gf_g/gf_c:4.1f}   checksum {checksum:.3e}"
    )

print("\nOK - if GPU GFLOP/s >> CPU, the pod is using the BC-250.")
