#!/usr/bin/env python3
"""
Cross-platform training + inference benchmark: ResNet-9 on CIFAR-10.

The same code runs on the BC-250 (--device ocl:0), a cloud GPU (--device cuda),
or a CPU (--device cpu). It writes one JSON result file per run; feed several to
compare.py for a side-by-side table. Protocol and rationale: BENCHMARK.md.

Self-contained: needs only torch (plus pytorch_ocl on the BC-250). It downloads
the CIFAR-10 python pickle itself - no torchvision.

  python benchmark.py --device ocl:0 --epochs 16 --out results/bc250.json
  python benchmark.py --device cuda  --epochs 16 --out results/t4.json
  python benchmark.py --device cuda  --epochs 16 --amp --out results/t4-amp.json
  python benchmark.py --device cpu   --epochs 4  --out results/cpu.json
"""
import argparse, contextlib, json, os, pickle, platform, statistics, tarfile, time, urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ device utils
def make_device(spec):
    if spec.startswith("ocl"):
        import pytorch_ocl  # noqa: F401  registers the "ocl" backend
    return torch.device(spec)

def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "ocl":
        torch.ocl.synchronize()

def dev_name(dev):
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    if dev.type == "ocl":
        return os.environ.get("BENCH_DEVICE_NAME", "OpenCL device ocl:0")
    return platform.processor() or platform.machine() or "CPU"

# --------------------------------------------------------- data (no torchvision)
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def load_cifar(root="./data"):
    os.makedirs(root, exist_ok=True)
    base = os.path.join(root, "cifar-10-batches-py")
    if not os.path.isdir(base):
        tgz = os.path.join(root, "cifar-10-python.tar.gz")
        if not os.path.isfile(tgz):
            print("downloading CIFAR-10 (~170 MB) ...")
            urllib.request.urlretrieve(CIFAR_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root, filter="data")

    def rd(fn):
        with open(os.path.join(base, fn), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], d[b"labels"]

    xs, ys = [], []
    for i in range(1, 6):
        a, b = rd(f"data_batch_{i}")
        xs.append(torch.tensor(a, dtype=torch.uint8))
        ys.extend(b)
    xtr = torch.cat(xs).reshape(-1, 3, 32, 32)
    ytr = torch.tensor(ys, dtype=torch.long)
    xte_raw, yte = rd("test_batch")
    xte = torch.tensor(xte_raw, dtype=torch.uint8).reshape(-1, 3, 32, 32)
    return (xtr, ytr), (xte, torch.tensor(yte, dtype=torch.long))

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

def augment_cpu(x):                       # x: uint8 [N,3,32,32] on CPU
    # Done on CPU with only trivially-portable ops so it can't hit a backend
    # op-coverage gap and is identical work on every platform.
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])       # horizontal flip (whole batch)
    x = F.pad(x, (4, 4, 4, 4))            # zero pad -> 40x40
    ox, oy = torch.randint(0, 9, (2,)).tolist()
    return x[:, :, oy:oy + 32, ox:ox + 32].contiguous()   # random 32x32 crop

# ------------------------------------------------------------- model: ResNet-9
def conv_bn(ci, co, pool=False):
    layers = [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(conv_bn(c, c), conv_bn(c, c))
    def forward(self, x):
        return x + self.conv(x)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(3, 64),
            conv_bn(64, 128, pool=True),
            Residual(128),
            conv_bn(128, 256, pool=True),
            conv_bn(256, 512, pool=True),
            Residual(512),
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.net(x) * 0.125        # output scaling from the fast-CIFAR recipe

# ---------------------------------------------------------------- eval / infer
@torch.no_grad()
def accuracy(model, x, y, dev, bs=512):
    model.eval()
    mean, std = MEAN.to(dev), STD.to(dev)
    correct = 0
    for i in range(0, x.size(0), bs):
        xb = x[i:i + bs].to(dev).float().div_(255).sub_(mean).div_(std)
        correct += (model(xb).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return correct / x.size(0)

@torch.no_grad()
def inference_bench(model, x, dev, sizes=(1, 8, 32, 128, 256), iters=50):
    model.eval()
    mean, std = MEAN.to(dev), STD.to(dev)
    out = {}
    for bs in sizes:
        if bs > x.size(0):
            continue
        xb = x[:bs].to(dev).float().div_(255).sub_(mean).div_(std)
        for _ in range(5):
            model(xb)
        sync(dev)
        lat = []
        for _ in range(iters):
            sync(dev); t0 = time.perf_counter()
            model(xb)
            sync(dev); lat.append((time.perf_counter() - t0) * 1e3)
        lat.sort()
        med = statistics.median(lat)
        out[f"bs{bs}"] = {
            "lat_ms_p50": round(med, 3),
            "lat_ms_p90": round(lat[int(len(lat) * 0.90)], 3),
            "lat_ms_p99": round(lat[int(len(lat) * 0.99)], 3),
            "img_per_s": round(bs / (med / 1e3), 1),
        }
    return out

# ---------------------------------------------------------------------- main
def run(a):
    dev = make_device(a.device)
    torch.manual_seed(a.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = a.amp   # TF32 off in the FP32 run
        torch.backends.cudnn.allow_tf32 = a.amp

    (xtr, ytr), (xte, yte) = load_cifar(a.data)
    mean, std = MEAN.to(dev), STD.to(dev)
    n = xtr.size(0)

    model = ResNet9().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    steps = a.epochs * ((n + a.batch_size - 1) // a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.25)
    lossfn = nn.CrossEntropyLoss()
    amp_ctx = (lambda: torch.autocast(dev.type)) if a.amp else contextlib.nullcontext

    hist, ep_sec, ep_ips = [], [], []
    print(f"device={dev}  name={dev_name(dev)}  torch={torch.__version__}  amp={a.amp}")
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        sync(dev); t0 = time.perf_counter()
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            xb = augment_cpu(xtr[idx]).to(dev).float().div_(255).sub_(mean).div_(std)
            yb = ytr[idx].to(dev)
            with amp_ctx():
                loss = lossfn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        sync(dev); dt = time.perf_counter() - t0
        acc = accuracy(model, xte, yte, dev)
        hist.append({"epoch": ep, "sec": round(dt, 2),
                     "img_per_s": round(n / dt, 1), "test_acc": round(acc, 4)})
        ep_sec.append(dt); ep_ips.append(n / dt)
        print(f"  epoch {ep:2d}  {dt:7.1f}s  {n/dt:8.0f} img/s  test_acc {acc:.4f}")

    warm = slice(1, None) if a.epochs > 1 else slice(None)   # drop epoch 0 (warm-up)
    result = {
        "workload": "resnet9-cifar10",
        "precision": "amp" if a.amp else "fp32",
        "device_arg": a.device,
        "device_name": dev_name(dev),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "train": {
            "median_epoch_sec": round(statistics.median(ep_sec[warm]), 2),
            "median_img_per_s": round(statistics.median(ep_ips[warm]), 1),
            "total_sec": round(sum(ep_sec), 1),
            "final_test_acc": hist[-1]["test_acc"],
            "epochs_to_0.90": next((h["epoch"] for h in hist if h["test_acc"] >= 0.90), None),
            "history": hist,
        },
        "inference": inference_bench(model, xte, dev),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", a.out)
    tr = result["train"]
    print(f"  train: {tr['median_img_per_s']:.0f} img/s  |  "
          f"{tr['median_epoch_sec']:.1f} s/epoch  |  "
          f"acc {tr['final_test_acc']:.3f}  |  ->90% at epoch {tr['epochs_to_0.90']}")
    if "bs1" in result["inference"]:
        print(f"  infer: bs1 {result['inference']['bs1']['lat_ms_p50']:.2f} ms  |  "
              f"bs128 {result['inference'].get('bs128', {}).get('img_per_s', 0):.0f} img/s")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", help="cpu | cuda | ocl:0")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.4, help="OneCycle max LR")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true",
                   help="autocast + TF32 (CUDA realistic best-case; leave off on ocl/cpu)")
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="results/bench.json")
    run(p.parse_args())
