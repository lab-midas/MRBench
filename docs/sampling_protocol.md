# Sampling protocol

Sampling has two modes.

## Retrospective

```text
complete dataset k-space
→ generated benchmark mask
→ mask broadcast over coil dimension and applied
→ undersampled ReconstructionInput
```

The sample normally has `mask=None`; a `SamplingProtocol`, requested
acceleration, and recorded seed are required. A generated mask is always
canonical `[S,T,Kx,Ky]`, without a coil dimension. Acceleration means only the
nominal acceleration requested from this benchmark mask.

`SamplingResult` records requested and actual acceleration, method, seed, and
parameters. Actual acceleration is grid locations divided by nonzero mask
locations when meaningful. Discrete masks can make it differ from the request.

Masks may be static or vary over cardiac time. Current public source-backed
selectors and implementation provenance are listed in
[`sampling_subsystem.md`](sampling_subsystem.md). The historical
`RandomCartesianSampling` class is internal test coverage and is not registered
as a current public selector.

## Acquired

```text
prospectively undersampled dataset k-space
+ acquired mask
→ ReconstructionInput
```

No sampling protocol is invoked and no second mask is generated. Missing masks
are an error. Actual acceleration is recorded from the acquired mask when it can
be calculated; requested acceleration and seed are unset.

## VISTA integration

For current in-house cine, Ky is phase encoding (canonical axis 3); VISTA never
subsamples Kx. Stored masks are `[Ky,T]`. They are transposed to `[T,Ky]` and
broadcast across slices and Kx to canonical `[S,T,Kx,Ky]`.

Generic `vista/precomputed` resolves one exact file at mask-generation time,
after the current subject's canonical geometry is known. Resolution precedence
is `sampling.parameters.mask_path` (explicit file), then `sampling.mask_root`
(explicit root), then the maintained default root
`/path/to/cmrbench-assets/masks`. Root-based selection constructs
`mask_VISTA_<Ky>x<T>_acc<R>_<seed>.txt`; `seed` is a file-selection identifier,
not a request to regenerate VISTA. An explicitly injected legacy generator is
still available only to direct tooling that supplies no file or root; it is not
a missing-file fallback for generic precomputed sampling.

`force_acs` defaults to false. In that ordinary mode, optional `ncalib` is
recorded but remains non-operative and the loaded VISTA mask is bitwise
unchanged. With `force_acs: true`, an explicit positive integer `ncalib<=Ky`
is required. CMRBench copies the base `[Ky,T]` mask and unions rows
`start=(Ky-ncalib)//2` through `stop=start+ncalib` (exclusive) across every T.
After transpose/broadcast this is a full-Kx centered Ky strip for every frame.
No base VISTA sample is removed and the source text file is never modified.

Provenance records the resolution mode, resolved root/filename/path, Ky, T,
native `[Ky,T]` and canonical `[S,T,Kx,Ky]` layouts, seed, forced-ACS bounds,
base/final hashes and sampled counts, added count, and base/final actual
acceleration. Requested R continues to select the base filename; public actual
R is recomputed from the final physical mask. A missing exact file or wrong
`[Ky,T]` shape is a hard failure; there is no nearest-R/geometry/seed choice,
resize, approximation, resampling-to-preserve-R, or generated fallback. VISTA
itself is not reimplemented.

Actual acceleration is recomputed from the final canonical mask. It remains
independent of requested/filename R and need not be numerically equal to it.
