# Benchmark: BC-250 vs cloud GPUs

Reproducible train-then-infer workload that runs identically on the
BC-250 (OpenCL) and on any CUDA cloud GPU.

## The workload

**ResNet-9 on CIFAR-10.** ~6.5M params, 16 epochs, batch 128.

- Every operation it uses (conv, batch-norm, ReLU, max-pool, linear, cross-entropy)
  is verified working on the BC-250's OpenCL path.
- Fits in <1 GB, so an 8 GB cloud card and the BC-250's 16 GB both run it at
  batch 128.
- FP32 is the only apples-to-apples precision (the BC-250 has no tensor cores
  and `pytorch_ocl` has no mixed precision). Cloud cards get a *second* run
  with `--amp` for their realistic best case.

## Results (2026-09-02)

BC-250 on `ocl:0` at two GPU-core configs (24 and 40 CUs enabled); cloud cards from Google Colab.

### Training

| Platform | Precision | Train img/s | s/epoch | Test acc | vs BC-250 40 CU |
| --- | --- | ---: | ---: | ---: | ---: |
| BC-250, CPU only | fp32 | 70 | 718 | 0.877 | 0.09× |
| BC-250, 24 CU | fp32 | 532 | 93.9 | 0.927 | 0.68× |
| **BC-250, 40 CU** | **fp32** | **786** | **63.6** | **0.923** | **1.0×** |
| T4 (Colab) | fp32 | 2,294 | 21.8 | 0.926 | 2.9× |
| T4 (Colab) | amp | 4,441 | 11.3 | 0.923 | 5.6× |
| L4 (Colab) | fp32 | 3,370 | 14.8 | 0.927 | 4.3× |
| L4 (Colab) | amp | 8,977 | 5.6 | 0.925 | 11× |
| A100 (Colab) | fp32 | 7,377 | 6.8 | 0.924 | 9.4× |
| A100 (Colab) | amp | 18,690 | 2.7 | 0.924 | 24× |

### Inference

`eval()` + `no_grad()`, batch 128, median of 50 timed iters after 5 warm-up.

| Platform | Precision | Infer img/s | vs BC-250 40 CU |
| --- | --- | ---: | ---: |
| BC-250, CPU only | fp32 | 214 | 0.06× |
| BC-250, 24 CU | fp32 | 2,586 | 0.66× |
| **BC-250, 40 CU** | **fp32** | **3,892** | **1.0×** |
| T4 (Colab) | fp32 | 7,217 | 1.9× |
| T4 (Colab) | amp | 7,355 | 1.9× |
| L4 (Colab) | fp32 | 12,119 | 3.1× |
| L4 (Colab) | amp | 18,321 | 4.7× |
| A100 (Colab) | fp32 | 21,865 | 5.6× |
| A100 (Colab) | amp | 49,612 | 13× |

### Notes

- 24 → 40 CUs scaled 532 → 786 img/s = **1.48× for a 1.67× core bump**
- Inference gap between BC-250 @ 40 CUs and T4 is only **~1.9×**
- Accuracy matches everywhere (0.922–0.927): the stack is numerically correct.
- @ 40 CUs / ~1.5 GHz the BC-250 is ~6–8 TFLOP/s FP32 → genuinely
  T4-class *silicon*. The ~3× FP32 gap is almost entirely **software
  stack** (kernel quality & maturity).

---

## Running it

### On the BC-250, through JupyterHub

1. Open `https://juphub.your-name.ts.net`, log in, start your server.
2. Get `benchmark.py` into your `/home/jovyan` (persistent). Easiest from the
   box:
   ```bash
   sudo kubectl -n jupyterhub cp benchmark.py  jupyter-<user>:/home/jovyan/benchmark.py
   sudo kubectl -n jupyterhub cp compare.py    jupyter-<user>:/home/jovyan/compare.py
   ```
   (or drag the file into the JupyterLab file browser, or paste it into a
   notebook cell prefixed with `%%writefile benchmark.py`).
3. In JupyterLab open a **Terminal** (Launcher → Terminal) and run:
   ```bash
   python benchmark.py --device ocl:0 --epochs 16 --out results/bc250.json
   python benchmark.py --device cpu   --epochs 4  --out results/bc250-cpu.json
   ```
   or from a notebook cell: `!python benchmark.py --device ocl:0 --epochs 16 --out results/bc250.json`
4. Results land in `~/results/*.json` on the persistent volume. Download them
   from the file browser (right-click → Download) to compare with cloud runs.

Note: Epoch 0 is slow (rusticl compiles kernels); the script already excludes it.

### On Google Colab (free T4)

1. [colab.research.google.com](https://colab.research.google.com) → New notebook.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Get `benchmark.py` in: click the folder icon in the left sidebar → Upload,
   pick `benchmark.py`. (Or paste it into a cell under a `%%writefile benchmark.py`
   line.)
4. Run, one per cell:
   ```
   !nvidia-smi
   !python benchmark.py --device cuda --epochs 16      --out results/colab-t4.json
   !python benchmark.py --device cuda --epochs 16 --amp --out results/colab-t4-amp.json
   ```
5. Pull the results down:
   ```python
   from google.colab import files
   files.download('results/colab-t4.json'); files.download('results/colab-t4-amp.json')
   ```

Note: Colab Pro (pay-as-you-go compute units) unlocks L4 / A100 the same
way.

### Compare

Drop every result JSON into this repo's `results/` directory and run:

```bash
python compare.py results/*.json
```

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
