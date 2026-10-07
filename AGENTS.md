# AGENTS.md

## Purpose

comfy-kitchen is a fast kernel library for diffusion inference (ComfyUI) with
multiple compute backends: eager (pure PyTorch), cuda, triton, hip (AMD
RDNA2-RDNA4), ascend (Huawei NPU). Python >= 3.10 (dev on 3.12), PyTorch >= 2.7.
Native CUDA/HIP extensions build through setup.py + CMake + nanobind; the
package imports and runs eager/triton without them.

## Layout

- `comfy_kitchen/` — public API (`import comfy_kitchen as ck`), `registry.py`
  (dispatch), `constraints.py`, `tensor/` (QuantizedTensor layouts)
- `comfy_kitchen/backends/{eager,cuda,hip,triton,ascend}/` — per-backend
  implementations; native sources live under `cuda/` and `hip/`
- `tests/` — pytest suite; `conftest.py` holds backend-availability gates
- `third_party/` — git submodules (cutlass, flash-attention), required for
  source builds
- `samples/` — standalone usage and benchmark scripts

## Conventions

- Dispatch is constraint-declared. Each backend registers FunctionConstraints
  (device, dtype, shape, compute capability) and the registry validates them
  before calling. Never wrap a kernel call in try/except for fallback. Input a
  kernel cannot take falls through to another capable backend, else
  `NoCapableBackendError`.
- Backend priority: hip -> cuda -> triton -> eager. Override per call
  (`backend=...`), per context (`ck.use_backend(...)`), or per process
  (`ck.set_backend_priority`, `registry.disable`).
- Tests that need a specific backend must gate on the helpers in
  `tests/conftest.py` (e.g. `requires_cuda_backend`) so they skip on CPU-only
  machines. `torch.cuda.is_available()` is not a CUDA-backend check: ROCm
  reports devices as "cuda". Tests must restore registry state; the autouse
  fixture does it, so do not depend on global changes persisting.
- `import comfy_kitchen` must keep working when no native extension is present.
- Style: ruff, line length 100, target py310, double quotes, isort. `N802` is
  ignored for ComfyUI node names. Docstrings in this codebase explain "why";
  keep that.
- Commits: `[Type][Backend] Summary`, e.g. `[BugFix][Ascend] ...`,
  `[Perf][HIP] ...`; release commits are `comfy-kitchen vX.Y.Z`.
- Contributions need a DCO sign-off (`git commit -s`) and the CLA comment on
  the PR (see CONTRIBUTING.md).

## Commands

    # Lint (this is what CI runs)
    ruff check .

    # Format (config in pyproject.toml; not enforced in CI)
    ruff format .

    # Tests (this is what CI runs)
    python -m pytest tests/ -v --tb=short

    # Subset / single test
    pytest tests/test_backends.py
    pytest tests/test_backends.py::TestBackendSystem::test_list_backends

    # Markers (pytest.ini / conftest.py): cuda, slow, cupy, performance

    # Dev install
    pip install -e ".[dev]"                    # builds native extensions
    pip install -e . --no-build-isolation -v   # faster rebuilds, sees local GPU

    # Submodules, required for source builds
    git submodule update --init --recursive

    # Build knobs (setup.py flags or env vars)
    COMFY_KITCHEN_BUILD_HIP=1 pip install .    # combined CUDA+HIP build
    COMFY_HIP_ARCHS=gfx1201 pip install .      # one HIP target, much faster
    python setup.py build_ext --cuda-archs="80;89" bdist_wheel
    # flags: --no-cuda --hip --no-hip --hip-archs=... --debug-build --lineinfo

## Constraints

- Architecture overrides fail closed. A target not in
  `comfy_kitchen/backends/hip/architectures.json` is rejected until its device
  and WMMA policies are reviewed and added. Do not add targets ad hoc.
- Both native extensions target the Python limited API (abi3) on 3.12+; keep
  that working. CMake >= 3.26 and Ninja are build requirements.
- No new hard dependencies: `dependencies = []` in pyproject.toml. Optional
  extras only (see `cublas`, `dev`, `build`).
- `MANIFEST.in` covers HIP native sources only; wheels ship the compiled
  extensions plus the package-data declared in pyproject.toml.
- Backend parity matters. CUDA work is routinely ported to HIP ("HIP port of
  ..."); when changing one backend's kernel, check whether the sibling backend
  needs the same fix.
- Kernel fallbacks must stay silent: unsupported dtypes/layouts go to another
  backend on the same device, never to CPU copies.
- Local `ruff check .` (0.12) reports two pre-existing UP038 findings in
  `comfy_kitchen/tensor/base.py`; current ruff removed that rule. Not yours to
  fix in unrelated work.