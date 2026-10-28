# Model adapter specification

## Sampling-condition separation

Adapters must not infer documented training exposure from runtime acceptance.
Physical `SamplingResult` provenance, model-side categorical/text/ACS inputs,
documented-reference matches, and observed runtime resolution are orthogonal.
SDUM and CardioMM may warn and run a valid OOD measurement; ambiguous automatic
semantic mappings require explicit model parameters. RAM remains categorical-
condition-free and consumes externally supplied sensitivity maps. Upstream
inference exceptions preserve their original type/message and are reported as
`model_inference`, not as physical sampling validation.

Every future adapter implements `ReconstructionModel`, receives a canonical
`ReconstructionInput`, and returns a `ReconstructionResult` in
`[S,T,X,Y]` geometry.

```python
class ExampleAdapter(ReconstructionModel):
    def load(self):
        ...

    def reconstruct(self, input):
        # canonical input
        # ↓
        # model-specific preprocessing
        # ↓
        # model inference
        # ↓
        # inverse preprocessing
        # ↓
        # canonical reconstruction
        ...

    def model_info(self):
        ...
```

An adapter may permute axes, convert tensor libraries, normalize, pad, crop,
coil-compress, estimate or convert sensitivity maps, build batches, move data to
a device, translate masks, or preserve a model's original FFT behavior. These
operations are local to the adapter and may not redesign the dataset or sampling
interfaces.

Significant transformations are returned as ordered `TransformRecord` entries,
for example axis permutation, 34-to-16 coil compression, padding to a model
matrix, or model-specific normalization. The adapter must not mutate the
canonical input in place.

`model_info()` supports model name/version, checkpoint and identifier, source
commit, parameter count, device, precision, and flexible additional metadata.
Fields may be unavailable in v0.1; future experiment records can populate them
without changing the base method.

`ZeroFilledAdapter` is the current sanity baseline. It applies `ifft2c` to
`[S,T,Kx,Ky,C]`, multiplies coil images by conjugated `[S,1,X,Y,C]` sensitivity
maps, and sums coils without a `sum(|smap|^2)` denominator. It returns
`[S,T,X,Y]` without normalization.

The manual full-data check is an **adjoint consistency check**, not an assertion
that the reference must be reproduced. With the established coil sum, its
prediction is `reference * sum(|sensitivity_maps|^2)`. The smoke test reports
both actual-versus-prediction consistency and actual-versus-reference discrepancy
without modifying or normalizing the maps.

The included SDUM adapter translates canonical input into its released private
layout, runs inference and restores `[S,T,X,Y]` while recording transforms and
provenance. RAM and CardioMM use the same public boundary with their own
released behavior. See [model setup](models.md) for their input requirements.
