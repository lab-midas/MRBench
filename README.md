# MRBench: Benchmarking Foundation Models for MRI Reconstruction

MRBench is a benchmark for evaluating foundation models for MRI reconstruction
across acquisition conditions, signal fidelities, and imaging domains.
It supports systematic comparisons of reconstruction quality.

This repository provides the MRBench Python reconstruction framework and
interactive dashboard.

## Framework Overview

![MRBench framework overview](website/assets/framework/Fig_1_Workflow.png)

## Python Framework

Install MRBench Core with Python 3.10 or newer, from the repository root:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e .
```

Run a small synthetic reconstruction and evaluation without pretrained weights:

```bash
python examples/make_toy_data.py
python -m cmrbench.reconstruct --config examples/toy_reconstruction.yaml
python -m cmrbench.evaluate --reconstruction-dir outputs/toy_reconstruction
```

For your own data, configure an experiment and the model runtime paths using
[getting started](docs/getting_started.md) and [model setup](docs/models.md), then run:

```bash
python -m cmrbench.run_experiment --config /path/to/experiment.yaml
```

Core dispatches reconstruction to the configured RAM, SDUM or CardioMM Python
interpreter and evaluation to the configured Core interpreter. Each model keeps
its own environment and preprocessing.

The RAM, SDUM and CardioMM **adapters are included**. Upstream model code and
pretrained weights are separate assets subject to their providers' terms.
CardioMM's author-provided checkpoint is confidential and is not distributed;
its adapter requires a separately authorized local checkpoint. CardioMM sampling
calls [externally installed upstream code](docs/models.md#cardiomm-sampling);
no adapted CardioMM sampling algorithms are bundled.

First-party licensing will be determined in November. Retain the
[third-party notices](THIRD_PARTY_NOTICES.md), including the
[VISTA research/nonprofit restrictions](cmrbench/vista/NOTICE.md).

## Interactive Dashboard

**[Explore the MRBench Dashboard →](https://lab-midas.github.io/MRBench/)**

Explore the framework, dataset and model information, experiment designs, available
results, and evaluation metrics and protocol. Published results currently cover
acceleration factor and sampling pattern; additional experiment results are planned.

## Repository Structure

```text
cmrbench/               Python framework and model adapters
configs/models/         Model configuration templates
examples/               Synthetic example and experiment templates
tests/                  Core tests
docs/                   Data, model setup and scientific protocols
licenses/               Third-party license text
pyproject.toml          Python installation metadata
THIRD_PARTY_NOTICES.md  Component attribution and terms
website/                Interactive dashboard source
  assets/               Original framework figure and website assets
  data/                 Published metrics, audit and resource metadata
  docs/                 Scientific protocol and maintenance documentation
  experiments/          Experiment views and comparison logic
.github/workflows/      GitHub Pages deployment and automated validation
```

## Models and Data

The benchmark integrates [RAM](https://github.com/matthieutrs/ram),
[SDUM](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI) and
[CardioMM](https://github.com/wangziblake/CardioMM_MMCMR-427K). Models, pretrained
weights and external datasets are credited to their original authors and providers.
The dashboard links to the original papers and code and identifies evaluated variants.

Published results use the in-house cardiac cine cohort. Dataset pages also introduce
CMRxRecon and fastMRI; benchmark results for these external datasets are not included
in this release. Local preview and validation instructions are in [website/README.md](website/README.md).
