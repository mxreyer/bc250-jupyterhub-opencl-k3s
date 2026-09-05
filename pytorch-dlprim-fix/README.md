# pytorch_dlprim fix: force the portable reduction path

**Problem:** on the BC-250, `pytorch_ocl` ([OpenCL backend for
PyTorch](https://github.com/artyom-beilis/pytorch_dlprim), built on
[DLPrimitives](https://github.com/artyom-beilis/dlprimitives)) fails to
compile any GPU kernel that reduces values across a work-group —
`softmax`, `log_softmax`, `cross_entropy`, `nll_loss`, the bias-gradient
step used by `Linear`/`Conv2d`, batchnorm sums, global pooling.  Plain
math (matmul, elementwise, convolution without bias) is unaffected.

**Root cause:** those kernels call the OpenCL 2.0 built-ins
`work_group_reduce_add()` / `work_group_reduce_max()`. rusticl only
allows programs to use those built-ins when it detects **Mesa's own
patched fork of libclc**
(`gitlab.freedesktop.org/karolherbst/mesa-libclc`). Fedora ships the
plain upstream libclc instead — that's the *"Patched Mesa libclc not
detected"* warning `clinfo`/rusticl prints on every run — so the
built-ins aren't declared and the kernel source fails with "use of
undeclared identifier".

**Fix:** DLPrimitives already has a fallback for exactly this. In
`dlprimitives/src/kernels/reduce.h`, an `#ifndef CUSTOM_REDUCE` block picks
between the OpenCL 2.0 built-ins and a hand-written reduction using
`__local` memory + `barrier()`, which needs nothing past OpenCL 1.2. That
fallback is off by default (`CUSTOM_REDUCE 0`). This directory forces it on.

There's no need to touch Mesa, libclc, or anything else on the system —
this is entirely a rebuild of one out-of-tree Python package inside its own
virtualenv.

## What's in this directory

- `custom_reduce.patch` — the one-hunk fix against `pytorch_dlprim`'s
  `dlprimitives` submodule (`src/kernels/reduce.h`).
- `pt_ocl.so` — a stripped copy of the rebuilt extension, checked in so the
  notebook image build (`k8s/jupyterhub/notebook-image/build.sh`) can pick it
  up without a rebuild. `build.sh` regenerates it in place.
- `build.sh` — reproduces the whole thing from a clean clone, in a
  self-contained venv under `./scratch/` (gitignored).

## How to rebuild from scratch

```
./build.sh
```

`build.sh` is self-contained:

1. Builds a throwaway Python 3.12 venv in `./scratch/venv/` with the same
   `torch==2.4.0` (CPU) and `pytorch_ocl` 0.2.0 wheel the notebook image
   uses. (Kept between runs; `rm -rf scratch/` to start clean.)
2. Clones `artyom-beilis/pytorch_dlprim` (with the `dlprimitives`
   submodule) into `./scratch/src/`, applies `custom_reduce.patch`.
3. Builds the extension and installs the stripped result over
   `./pt_ocl.so` next to the script — which is what
   `k8s/jupyterhub/notebook-image/build.sh` copies into the image.

To try the rebuild in the scratch venv directly (matmul / softmax checks):

```
cp scratch/build/pytorch_ocl/pt_ocl.so \
   scratch/venv/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so
rm -f ~/.dlprimitives/cache.db   # old cache keyed to the old kernel source
```

Requires `python3.12`, `python3.12-devel`, `cmake`, `git`, `sqlite-devel`,
`OpenCL-ICD-Loader-devel`.

## Reverting

The stock (unpatched) extension is whatever the `pytorch_ocl` 0.2.0 wheel
ships. To go back, reinstall it into the venv and delete the kernel cache:

```
scratch/venv/bin/pip install --force-reinstall --no-deps \
  "https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"
rm -f ~/.dlprimitives/cache.db
```

For the notebook image, drop the `CUSTOM_REDUCE` `COPY`/`RUN` lines from
`k8s/jupyterhub/notebook-image/Dockerfile` and rebuild.

---

*Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5).*
