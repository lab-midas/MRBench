# fastMRI multi-coil input preparation

This capability prepares static fastMRI slices without changing the generic
reconstruction entry, `SamplingProtocol`/`SamplingResult`, `ReconstructionInput`,
or any model adapter.  It does not run a reconstruction model.

## Data and geometry

The low-level `cmrbench.data.fastmri.load_fastmri_multicoil_slice` reads one selected
HDF5 slice from `kspace` and `reconstruction_rss`.  Raw complex k-space is
official fastMRI `[C,H,W]`.  The RSS array supplies the reference spatial shape
and an optional magnitude reference; it is not used as a complex model input.

Before sampling, every coil is transformed with a centered orthonormal IFFT,
center-cropped identically in the complex image domain, and transformed back
with the inverse centered orthonormal FFT convention.  There is no resize,
RSS/VCC, coil compression, magnitude conversion, or intensity normalization.
The resulting mappings are:

- k-space `[C,H,W] -> [1,1,H,W,C]`;
- reference `[H,W] -> [1,1,H,W]`.

The generic dataset wrapper also supports `dataset.selection.slices: all` for
one volume, preserving selected slices as `S` with `T=1`. Selecting one integer
slice uses the low-level mapping above. A configuration template is in
[`../examples/fastmri_ram.yaml`](../examples/fastmri_ram.yaml); external data and
the model/calibration runtimes are still required.

`build_fastmri_multicoil_slice` returns an unsampled `CineMRISample`.  The
existing `prepare_reconstruction_input` must then realize the physical mask and
broadcast it across coils.  Only after that step may
`attach_prepared_sensitivity_maps` run.

## Sensitivity-map strategies

Dataset parameters use `sensitivity_maps.source`; `strategy` is accepted as an
equivalent programmatic spelling.  The complete example is in
[`examples/fastmri_ram.yaml`](../examples/fastmri_ram.yaml).

`provided` validates canonical `[S,1,X,Y,C]` or `[S,T,X,Y,C]` shape, slice/time,
spatial geometry, coil count, complex dtype, and finite values.  Values,
normalization, phase, coil ordering, and coil basis are not modified.

`espirit` uses direct SigPy `EspiritCalib` on CPU.  The input-preparation extra
pins `sigpy==0.1.27`; it is intentionally independent of the RAM runtime and
DeepInv.  Defaults mirror SigPy's public API: kernel width 6, calibration-matrix
threshold (`thresh`) 0.02, eigenvalue crop threshold (`crop`) 0.95, and 100
iterations.  An explicit
`calib_width` may be requested. At the low-level estimator API, `null` uses
the detected realized ACS extent; formal generic experiments do not rely on
that fallback. They resolve the requested width from physical sampling
(`ncalib`, or `round(runtime Ky * center_fraction)`) before calling ESPIRiT.
An explicitly configured `calib_width` must match that requested width.
The realized ACS extent remains a feasibility check and separate provenance,
not the default generic calibration policy. The current CMRBench fastMRI
benchmark's `ncalib=24` is a chosen acquisition setting, not a fastMRI format
requirement.

ACS is detected from the actual canonical `[S,T,Kx,Ky]` mask.  A Ky line is
eligible only if every Kx location is acquired across all represented slices
and frames.  The detector selects the largest contiguous strip that contains
the center (both center-adjacent lines for even Ky).  Missing central support is
an error.  Fully sampled lines outside this contiguous strip are ignored.

The estimator receives a freshly zeroed calibration array containing only the
selected ACS copied from the already masked/acquired k-space.  It can never use
unacquired samples from the original full source, even if a caller accidentally
passes nonzero values outside the realized mask.  ESPIRiT currently accepts
only static `T=1`; output is canonical `[S,1,X,Y,C]`.

Provenance is stored under `ReconstructionInput.metadata` key
`cmrbench_sensitivity_maps`.  It includes source, backend and version, detected
ACS bounds/extent, requested parameters, actual calibration region, kernel,
crop and threshold values, input/output shapes, coil count, prewhitening state,
and exact normalization/phase/basis handling.

## Required ordering

For fully sampled fastMRI train/validation files:

1. load one raw multi-coil slice;
2. image-domain center crop to reference geometry;
3. map to canonical full k-space;
4. call the existing sampler and apply its realized mask;
5. call `attach_prepared_sensitivity_maps` with `provided` or `espirit`;
6. pass the unchanged `ReconstructionInput` contract to RAM.

Estimating maps before retrospective sampling is not a supported default.
SDUM and CardioMM are unchanged and may continue to ignore externally attached
maps according to their existing adapters.
