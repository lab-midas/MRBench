# Canonical data specification

## `CineMRISample`

`CineMRISample` is the dataset boundary. Its fields are:

```text
kspace:           [S,T,Kx,Ky,C]
mask:             [S,T,Kx,Ky] or None
reference:        [S,T,X,Y] or None
sensitivity_maps: [S,1,X,Y,C] or supported [S,T,X,Y,C]
metadata:        sample/cohort provenance
```

`CineMRISample.kspace` may contain either complete or prospectively
undersampled measurements. For complete benchmark starting data, `mask=None` is
preferred. For acquired undersampled data, `mask` is the actual acquisition
mask and must not be discarded.

K-space is expected to be complex-valued. Version 0.1 does not force a specific
tensor library or complex precision; adapters may convert privately and record
that conversion.

The loader performs no model-specific resizing, normalization, cropping, axis
permutation, or coil compression. Subjects may have different spatial sizes and
coil counts. A reference and sensitivity maps are both optional.

For current in-house cine, X/Kx is frequency encoding and Y/Ky is phase
encoding. Trusted time-averaged sensitivity maps use `[S,1,X,Y,C]` and
broadcast over T. A future dataset may provide `[S,T,X,Y,C]`.

The in-house reader loads the stored `[S,T,X,Y,1]` normalized coil-combined
reference and sensitivity maps, multiplies them to obtain coil images, and uses
`fft2c` to synthesize complete benchmark k-space. This is starting k-space for
retrospective evaluation, not raw acquired scanner k-space.

## `ReconstructionInput`

`ReconstructionInput` is the model boundary:

```text
kspace: [S,T,Kx,Ky,C]
mask:   [S,T,Kx,Ky]
```

Unlike the sample, it always describes actual measurements given to the model
and therefore has an explicit coil-independent mask. Retrospective preparation
generates and applies the mask. Acquired preparation uses dataset k-space and
mask directly. An optional full-data/R=1 experiment can generate an all-ones
mask.

For the full-data diagnostic, the established sensitivity-weighted adjoint gives
approximately `reference * sum(|sensitivity_maps|^2, coil)`. The server smoke
test therefore calls this an adjoint consistency check and separately reports
the discrepancy from the original reference. It does not divide by map power.

Model adapters must adapt to the canonical representation rather than changing
the canonical representation to match the model. Permutations, padding,
normalization, coil compression, device movement, and inverse transformations
belong inside the adapter and should be recorded as `TransformRecord` entries.

Validation is intentionally lightweight: k-space must be five-dimensional,
masks exactly match its first four dimensions, and a reference must match slice
and time dimensions. Spatial matrix sizes are not compared across subjects.
