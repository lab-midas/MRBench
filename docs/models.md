# Model runtimes and assets

MRBench Core provides interfaces, orchestration and evaluation. It does not
install pretrained networks or their dependency stacks. Use separate upstream
runtimes and retain the existing backend/variant settings.

## Runtime executables

| Selection | Environment variable |
| --- | --- |
| RAM `implementation: official_github` | `CMRBENCH_RAM_OFFICIAL_PYTHON` |
| RAM `implementation: deepinv` | `CMRBENCH_RAM_DEEPINV_PYTHON` |
| SDUM `variant: small\|base\|large` | `CMRBENCH_SDUM_PYTHON` |
| CardioMM | `CMRBENCH_CARDIOMM_PYTHON` |
| Evaluation, when enabled | `CMRBENCH_EVALUATION_PYTHON` |

Set each variable to an absolute Python executable path before starting
`cmrbench.run_experiment`. Pass one executable, without embedded quotes,
shell arguments, a Conda environment name or `conda run`. Paths containing spaces
are supported as one argument. Executable strings are not expanded or normalized;
avoid `~` and literal shell-variable expressions.

An unset variable defaults to `sys.executable`, captured at import time.
The fallback does not check installed model dependencies or switch environments.
An explicitly configured `model.environment` must match the selected string
exactly. Prefer leaving this optional field out and configuring executables once
through the variables above.

The orchestrator passes the resolved experiment-config path to
`python -m cmrbench.reconstruct`, then the resolved output directory to
`python -m cmrbench.evaluate`. Both children inherit the working directory
and environment. See [commands and failure handling](getting_started.md).

## Prepare separate model environments

Use the upstream project's requirements at the pinned revision for the chosen
backend. RAM's two DeepInv versions belong in separate environments; all three
SDUM variants use the SDUM runtime. Match PyTorch, CUDA and the GPU to the
upstream stack. Do not install all model stacks into Core.

Every runtime must import this same `cmrbench` checkout and its required core
modules. For compatible Python >=3.10 runtimes, an editable installation is
available; check dependency compatibility before changing a frozen model stack.
A source checkout on `PYTHONPATH` is the existing alternative for legacy
runtimes. Configure that path explicitly in the child process environment.
No model environment is created or installed by the orchestrator.

The historical CardioMM runtime used Python 3.8.20, PyTorch 2.0.1+cu117,
NumPy 1.24.4 and Transformers 4.30.2. It uses the source/PYTHONPATH route because
Core's installation metadata requires Python >=3.10. Keep evaluation in the core
environment. Runtime dispatch tests do not validate these pretrained stacks;
real source/checkpoint/CUDA execution requires a separate validation.

## RAM

Select a backend through the existing model YAML:

| YAML | Backend and assets |
| --- | --- |
| `configs/models/ram_deepinv.yaml` | DeepInv 0.3.5; `deepinv.models.RAM(pretrained=True)` |
| `configs/models/ram.yaml` | Pinned external RAM `0de1617403291e5a28179097795752200e65b82c`; DeepInv 0.3.0; `ram_source_root` and optional local `checkpoint` |

The DeepInv backend does **not** accept a user-specified `checkpoint` argument.
Leave its YAML `checkpoint: null`. Its public constructor resolves pretrained
assets and may download on first use; prepare its expected cache for offline
execution. The adapter verifies the expected artifact and loaded state.

The official-GitHub backend permits a local checkpoint or its existing official
cache/download path. The two backends remain distinct even when checkpoint
identities match. Keep the pinned external source checkout available for revision
verification. Upstream: [RAM](https://github.com/matthieutrs/ram) and
[DeepInv](https://github.com/deepinv/deepinv).

RAM reconstructs independent frames with complex sensitivity maps. Prepared cine
provides maps; static fastMRI can estimate them from acquired ACS. The adapter
adds no external intensity scaling or measurement noise.

## SDUM

Choose `sdum_small.yaml`, `sdum_base.yaml` or `sdum_large.yaml` under
`configs/models/`. Set `sdum_source_root`, the local `checkpoint`, and its
matching JSON `config`. All variants use `CMRBENCH_SDUM_PYTHON`.

The source revision is `99be712b2e9fd34bf6a2f709373d311354fbda96`.
Retain the upstream PyTorch/MONAI and other model dependencies.
Upstream: [NV-Raw2Insights-MRI](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI).

The adapter preserves cyclic five-frame windows, internal sensitivity estimation,
CUDA BF16 execution and RSS magnitude output. Base/Large use acquisition and
sampling conditioning. External maps do not replace the internal estimator.
Configure physical sampling and model conditioning separately; runtime acceptance
does not establish training exposure.

## CardioMM

The adapter is available, but its author-provided reconstruction weights are
**confidential and are not provided or redistributed**. Separately authorized
users may point `configs/models/cardiomm.yaml` to their own local checkpoint.
No public checkpoint access is implied.

Configure `source_root` at `abb6f8a4e293426736467c9859a5f5755c38c8aa`,
the authorized local `checkpoint`, and `clip_dir`. The pinned loader expects
CLIP at `<source_root>/clip-vit-base-patch16`; these paths must agree.
Upstream: [CardioMM](https://github.com/wangziblake/CardioMM_MMCMR-427K).

Supply actual `lifespan`, `field_strength`, `vendor`, `scanner`, `modality`
and `view` values. The YAML's acquisition values are examples from the original
study. Preserve batch size 1, cached text conditioning and `no_grad`.
The adapter reconstructs frames independently with internal sensitivity maps,
retaining the released model's preprocessing.

## CardioMM sampling

CardioMM sampling algorithms are not bundled. Obtain the original public
[CardioMM source](https://github.com/wangziblake/CardioMM_MMCMR-427K/tree/abb6f8a4e293426736467c9859a5f5755c38c8aa)
at revision `abb6f8a4e293426736467c9859a5f5755c38c8aa` under its academic-use
terms. For any model using `sampling.implementation: cardiomm`, add both the
upstream repository root and its
`CardioMM_dataundersampling/Python_Reimpl_Version` directory to that model
interpreter's `PYTHONPATH`. Use your platform's path separator (`:` on POSIX,
`;` on Windows). This is separate from selecting the CardioMM model adapter.

The four uniform/k-t generators require NumPy, SciPy and scikit-image. The
training-side `random` and `radial` generators also import the upstream
reconstruction module, which requires PyTorch and fastMRI in that runtime.
MRBench checks the called source files against pinned SHA-256 fingerprints
(normalizing Windows line endings) and fails if the code is missing or differs.
Keep the entire upstream checkout at the pinned revision, including its helpers.
There is no automatic download, copied algorithm or alternative-sampler fallback.
Tests exercising CardioMM sampling require the same external source and dependencies.
The existing canonical layout, ACS validation, radial acceleration scale and
explicit Gaussian/Python random seed policies are retained.

## Evidence and terms

Keep nominal/actual acceleration, physical mask identity, model conditioning,
adapter/source versions, checkpoint identifiers and runtime versions with each
run. Document shifted sampling conditions explicitly. See the
[evaluation protocol](evaluation_protocol.md).

The frozen Dashboard cohort uses RAM DeepInv, SDUM Base and CardioMM.
Other available variants are capabilities, not additional models in that table.
Model-specific execution in a new environment remains to be validated.
First-party licensing will be determined in November; preserve [existing notices](../THIRD_PARTY_NOTICES.md) and
[VISTA restrictions](../cmrbench/vista/NOTICE.md).
