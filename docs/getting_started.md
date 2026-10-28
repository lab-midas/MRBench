# Running a reconstruction

## Synthetic first run

Use the installation and synthetic commands in the [README](../README.md) from
the repository root. The toy generator creates a moving complex ring and four
sensitivity maps. Its zero-filled scores demonstrate the pipeline, not pretrained
model performance.

`outputs/toy_reconstruction/` contains `config.yaml`,
`experiment_metadata.json`, `results.csv` and `subjects/` with reconstruction
arrays and metadata. Evaluation adds `evaluation/metrics_v2.csv`,
`metrics_v2.json`, `evaluation_metadata.json` and QC/animation assets.
Evaluation still needs the configured source arrays to reload the reference.

## Configure your data and model

Use `examples/cine_ram.yaml` for prepared cine or
`examples/fastmri_ram.yaml` for static multi-coil fastMRI. Replace their data
placeholders and choose an existing model YAML through `model.config`:

```yaml
model:
  config: ../configs/models/ram_deepinv.yaml
```

The model reference resolves relative to the experiment YAML. Data, asset and
output paths resolve against the process working directory; run from the
repository root. YAML loading does not interpolate environment variables; use
explicit paths rather than relying on tilde expansion. Use platform paths,
such as `'C:/data/cine/preprocess'` in Windows YAML.

Change the model reference to an SDUM variant or CardioMM YAML only after
configuring its assets and acquisition metadata in [model setup](models.md).
Keep the physical sampler's `implementation` explicit.

## Dispatch from MRBench Core

Set the model and evaluation executable variables before launching Python.
For example, for RAM DeepInv in a POSIX shell:

```bash
export CMRBENCH_RAM_DEEPINV_PYTHON=/path/to/ram-runtime/bin/python
export CMRBENCH_EVALUATION_PYTHON=/path/to/core-runtime/bin/python
python -m cmrbench.run_experiment --config examples/cine_ram.yaml
```

Or in PowerShell:

```powershell
$env:CMRBENCH_RAM_DEEPINV_PYTHON = 'C:/path/to/ram-runtime/Scripts/python.exe'
$env:CMRBENCH_EVALUATION_PYTHON = 'C:/path/to/core-runtime/Scripts/python.exe'
python -m cmrbench.run_experiment --config examples/cine_ram.yaml
```

The [runtime table](models.md#runtime-executables) lists all five variables.
Their values are read when the runtime module is imported. An unset variable
falls back to the launching Python executable; this succeeds only when that
interpreter already has the selected model's dependencies and assets.
Set the variables explicitly for separate environments.

Each child must import the same MRBench checkout through its own installation
or an absolute `PYTHONPATH` entry. No Conda activation or environment creation
is performed. An optional YAML `model.environment` must exactly match the
selected executable string; it is a consistency check, not a runtime override.
Omitting it avoids duplicate path configuration.

Reconstruction is followed by evaluation unless `evaluation.enabled: false`.
The zero-filled toy uses the separate commands in the README; the orchestrator's
runtime selector covers RAM, SDUM and CardioMM.

## Run stages separately

In the selected model interpreter:

```bash
/path/to/model-python -m cmrbench.reconstruct --config examples/cine_ram.yaml
```

In the core/evaluation interpreter:

```bash
/path/to/core-python -m cmrbench.evaluate --reconstruction-dir outputs/cine_ram_example
```

`cmrbench.reconstruct` does not select or launch another interpreter. The
experiment's `output.root` determines reconstruction storage in both workflows.

## Failures and short inputs

A missing interpreter fails orchestration preflight. A reconstruction failure
stops evaluation; an evaluation failure preserves saved reconstructions.
Inspect `experiment_run_metadata.json` for the failed stage and exception.
A child that exits successfully with incomplete outputs also fails verification.

Replace every `/path/to/...` placeholder before pretrained use. If sampling
conditioning is rejected, check the physical mask, calibration and model labels.
Default QC phases `[6,22]` are zero-based and require at least 23 phases;
use `[0]` and disable cine GIFs for static data.
