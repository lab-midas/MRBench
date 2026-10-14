# CMRBench reconstruction evaluation protocols

## Versioned reporting policy

CMRBench reconstruction reports must name an immutable protocol version. The
current policy is:

- **Primary:** `cmrbench_recon_metrics_v2_subject_p995`, selected as
  `v2_subject_p995`. This is the subject-normalized cross-model reconstruction
  protocol for new benchmark evaluation.
- **Secondary diagnostic:** `cmrbench_recon_metrics_v1`, selected as
  `v1_raw`. This is the preserved raw-magnitude, amplitude-sensitive protocol.

Historical v1 JSON/CSV artifacts remain valid and must not be overwritten or
silently reinterpreted. CMRBench v2 is a deliberate project convention, not a
claim of a universal MRI reconstruction standard.

## Metrics v2: primary subject-normalized protocol

### Magnitude domain and one scale per complete subject source

For canonical subject volumes `[S,T,X,Y]`:

```text
R = abs(reference)
Z = abs(zero_filled)
X = abs(reconstruction_candidate)

s_R = P99.5(R over every S,T,X,Y value)
s_Z = P99.5(Z over every S,T,X,Y value)
s_X = P99.5(X over every S,T,X,Y value)

R_n = R / s_R
Z_n = Z / s_Z
X_n = X / s_X
```

Each source is normalized independently. There is exactly one scalar per
subject per image source—not per slice, phase, frame, pixel, or coil. Candidate
scales do not use the reference, and no least-squares/reference-fitted gain is
estimated.

Each p99.5 scale must be positive and finite. An all-zero or nonfinite source
is an explicit invalid evaluation condition. No epsilon is added and no silent
repair is permitted.

### No scientific clipping

Normalized arrays are **not** clipped to `[0,1]`. Values such as `1.05`, `1.4`,
or `3.0` remain in the metric inputs. This preserves penalties from isolated
bright artifacts, local overshoot, and other signal above the robust subject
scale. P99.5 is used instead of the maximum so one extreme pixel does not set
the complete cine's scale.

### Frame metrics

After subject normalization, metrics are evaluated independently for every 2-D
`(slice, phase)` frame.

```text
NRMSE_v2 = ||X_n - R_n||_2 / ||R_n||_2

MSE_v2   = mean((X_n - R_n)^2)
PSNR_v2  = 10 * log10(1.0^2 / MSE_v2)
```

SSIM is:

```python
skimage.metrics.structural_similarity(
    R_n,
    X_n,
    data_range=1.0,
    win_size=7,
    gaussian_weights=False,
)
```

PSNR and SSIM always use `data_range=1.0`, including when normalized arrays
contain values above one. No candidate-derived or rare-outlier-derived data
range is substituted.

ZeroFilled uses the identical v2 evaluator: its complete subject magnitude is
divided by its own `s_Z`, then compared frame-by-frame with `R_n`.

### Subject and cohort aggregation

Every frame-level value is retained. The subject endpoint is the equal
arithmetic mean over all `S*T` frames. Pixels are not flattened into one
subject metric, and slices/phases are not differentially weighted. Planned
cohort aggregation remains:

```text
frame metrics -> equal frame mean within subject -> equal-weight subject summary
```

### Required amplitude-scale diagnostics

Every v2 result records:

- `reference_p99_5`;
- `zero_filled_p99_5`;
- `candidate_p99_5`;
- `zero_filled_to_reference_p99_5_ratio`;
- `candidate_to_reference_p99_5_ratio`.

`normalized_max` and `fraction_of_values_above_1` may also be recorded as QA.
These scale fields are diagnostics, not ranking metrics.

## Metrics v1: preserved raw amplitude-sensitive diagnostic

Metrics v1 remains numerically unchanged:

```text
r = abs(reference_frame)
c = abs(candidate_frame)

NRMSE_v1 = ||c-r||_2 / ||r||_2

MSE        = mean((c-r)^2)
data_range = max(r)-min(r)
PSNR_v1    = 10*log10(data_range^2/MSE)
```

SSIM v1 uses the same raw reference-derived `data_range`, `win_size=7`, and
`gaussian_weights=False`. It applies no normalization, gain fitting, histogram
matching, or clipping. It remains valuable for exposing absolute-amplitude
discrepancies, but it is no longer the primary cross-model reconstruction
ranking protocol.

The legacy functions `compute_frame_metrics(...)` and calls to
`compute_cine_frame_metrics(...)` without a protocol argument retain v1
semantics for backward compatibility. New evaluations must pass
`protocol="v2_subject_p995"` explicitly.

## Why the primary protocol changed

Pretrained reconstruction models may emit the same spatial reconstruction
under different positive global output-intensity conventions. A global scale
should not dominate the primary cross-model anatomical reconstruction result.
Independent full-subject p99.5 normalization removes that one global gain while
remaining robust to an isolated maximum. The absence of clipping still
penalizes localized high-intensity errors. Raw v1 remains alongside v2 so
absolute-amplitude fidelity is never hidden.

## QC v1 science remains frozen and complementary

This visualization refinement does not change QC error definitions or image
scaling semantics.

- Anatomy QC independently autoscales panels and displays scale-invariant
  structural error.
- Fidelity QC uses the raw reference-derived display window and shows
  amplitude-sensitive error.

QC display saturation does not clip scientific metric arrays. Metrics v2
normalization must not be applied to QC v1 figures.

Generic Anatomy and Fidelity figures use the Matplotlib `turbo` error-map
colormap with an unlabeled colorbar. Each selected phase is a 2x3 block:
Reference/Zero-filled/Reconstruction above the verified canonical sampling
mask/ZF Error map/Reconstruction Error map. Phase blocks are arranged
side-by-side. The mask is selected as `canonical_mask[s,t,:,:]`; its concise
title reports the already validated global complete-mask actual R, never a
new per-frame acceleration. Anatomy errors are display-saturated with
`vmin=0,vmax=0.2`; Fidelity errors retain `vmin=0,vmax=1`. Neither scientific
error array is numerically clipped.

### Formal generic QC frame selection

Generic evaluation uses `mid_slice_fixed_phases_v1`. For the actual canonical
evaluated shape `[S,T,X,Y]`, it selects `(S//2,6)` and `(S//2,22)`; both phase
indices are zero-based. The selector never reads a manifest `SLICE_DIM`. If
`T<=22`, evaluation fails unless the experiment explicitly supplies valid
`evaluation.qc.phase_indices`; it does not clamp or revert to the historical
first/middle/last display policy. This changes displayed frames only: Metrics
v2 still consumes every one of the `S*T` frames, and the frozen Anatomy and
Fidelity QC computations and image-scaling semantics are unchanged.

When enabled, the optional candidate-only
`<subject>_mid_slice_reconstruction.gif` displays
`abs(reconstruction[S//2,0:T])`. One linear `[0,max(abs(mid-slice cine))]`
window is fixed across every temporal frame. This visualization neither
modifies the saved candidate nor participates in any metric.

## Reference and large-array policy

The benchmark target remains `abs(canonical reference)`. The informational
complete-multicoil RSS sanity audit does not replace it, and no SDUM-specific
target is introduced. Saved RAM/SDUM reconstructions stay server-side; metrics
scripts read them there and produce new versioned JSON/CSV artifacts.

## Implementation

- Versioned common metrics:
  `cmrbench/evaluation/reconstruction_metrics.py`
- Frozen common QC:
  `cmrbench/evaluation/reconstruction_qc.py`
- Generic evaluation entry point and formal frame selector:
  `cmrbench/evaluate.py`
- Candidate-only fixed-window GIF:
  `cmrbench/evaluation/reconstruction_gif.py`
- RAM saved-output reevaluation:
  `scripts/evaluate_reconstruction_metrics.py`
- SDUM Phase-3C saved-output reevaluation:
  `scripts/sdum_phase3c_evaluate_r8.py`

Metrics v2 outputs use `_v2` filenames and fresh output directories. Existing
Metrics v1 JSON/CSV/PNG files are not overwritten.
