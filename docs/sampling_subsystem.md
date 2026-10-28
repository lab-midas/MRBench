# Sampling subsystem

The generic physical-mask contract does **not** require a fully sampled central
ACS. ACS is detected and serialized; source generators that promise an ACS keep
their own hard postcondition, while SDUM/CardioMM sensitivity-estimation
semantics stay model/runtime-specific. See
[the public model compatibility guide](models.md)
for model-facing calibration requirements and the conditioning boundary.

CMRBench separates a sampling family (`sampling.method`) from the concrete
generator (`sampling.implementation`). Model selection never chooses or rewrites
the generator. `prepare_reconstruction_input()` applies the returned Boolean
mask once to complete canonical k-space `[S,T,Kx,Ky,C]`; every protocol returns
`SamplingResult.mask` as `[S,T,Kx,Ky]`.

CardioMM selectors require [external upstream sampling code](models.md#cardiomm-sampling).
The registry and scientific settings are unchanged; no copied CardioMM algorithm is bundled.

## Public selector contract

| method | implementation | implementation rule | native construction | ACS |
|---|---|---|---|---|
| `uniform` | `canonical` by default; `sdum` or `cardiomm` only at the documented centering boundary | omit selector in the equivalence domain; explicitly select source for even Ky + odd `ncalib` | static `[Kx,Ky]`, repeated over T, then S | centered full-Kx Ky strip, configurable `ncalib` (default 20) |
| `kt_uniform` | `sdum` or `cardiomm` | required | joint `[Ky,T]`, expanded over Kx, then S | centered full-Kx Ky strip |
| `kt_gaussian` | `sdum` or `cardiomm` | required; explicit seed required | joint `[Ky,T]`, expanded over Kx, then S | centered full-Kx Ky strip |
| `kt_radial` | `sdum` or `cardiomm` | required | rasterized `[Kx,Ky,T]`, then S | centered square |
| `random` | `sdum` or `cardiomm` | required | NVIDIA `[Ky]` center-fraction random, or CardioMM fixed-ACS `[Kx,Ky]`; then S/T broadcast | centered full-Kx Ky strip |
| `equispaced` | `sdum` | optional; resolves to `sdum` | NVIDIA-modified fastMRI `[Ky]`, then S/T/Kx broadcast | source-faithful `center_fraction` |
| `radial` | `cardiomm` | optional; resolves to `cardiomm` | CardioMM training-side static `[Kx,Ky]`, then S/T broadcast | centered square `ncalib` |
| `vista` | `precomputed` (default) or `cmrbench` | omitted legacy selector resolves to `precomputed`; otherwise explicit | disk-backed or native compact `[Ky,T]`, shared post-generation ACS handling, expanded over Kx and S | none by default; explicit centered Ky strip with `force_acs` |
| `external_mask` | caller value or `external` | caller-supplied | explicit declared layout only | none inserted |

Missing implementation on legacy `kt_*` YAML temporarily resolves to `sdum`
with `FutureWarning`. Checked-in configs are explicit. The effective `config.yaml`
written for a run contains the resolved implementation. Unknown implementation
values fail closed.

The effective public inventory has nine methods and sixteen first-class paths:
`uniform/{canonical,sdum,cardiomm}`, `kt_uniform/{sdum,cardiomm}`,
`kt_gaussian/{sdum,cardiomm}`, `kt_radial/{sdum,cardiomm}`,
`random/{sdum,cardiomm}`, `equispaced/sdum`, `radial/cardiomm`,
`vista/{precomputed,cmrbench}`, and `external_mask/external`. The registry also
retains VISTA's method-level default factory so omitted legacy selectors keep
the established `precomputed` behavior.

## Constructor and YAML parameter source of truth

All retrospective configurations require top-level `mode`, `method`, and
`acceleration`. The exact implementation-specific contract is:

| path | consumed constructor parameters | defaults | seed behavior |
|---|---|---|---|
| `uniform/{canonical,sdum,cardiomm}` | `ncalib` | 20 | optional provenance only; generator is deterministic |
| `kt_uniform/{sdum,cardiomm}` | `ncalib` | 20 | optional provenance only; generator is deterministic |
| `kt_gaussian/{sdum,cardiomm}` | `ncalib`, `alpha` | 20, 0.2 | required and consumed |
| `kt_radial/{sdum,cardiomm}` | `ncalib`, `angle4next`, `cropcorner`, `radial_acceleration_scale` | 20, 137.5, true, 0.6 | optional provenance only; generator is deterministic |
| `random/sdum` | `center_fraction` | none; required | required and consumed |
| `random/cardiomm` | `ncalib` | 20 | required and consumed |
| `equispaced/sdum` | `center_fraction` | none; required | required random offset |
| `radial/cardiomm` | `ncalib` | 20 | required initial-angle draw; CMRBench explicitly seeds the source behavior |
| `vista/precomputed` | optional `mask_path`, `mask_root`, `force_acs`, `ncalib`, `vista_type` | default root below; `force_acs=false`; ncalib absent; `VISTA` | required file-selection identifier; does not generate |
| `vista/cmrbench` | optional `force_acs`, `ncalib` | `force_acs=false`; ncalib absent | required and passed unchanged to the native generator |
| `external_mask/external` | exactly one of programmatic `mask` or `.npy` `mask_path`; `layout`, `source`, inner provenance `parameters` | `layout=S,T,Kx,Ky`, `source=external` | optional provenance only |

The shared NVIDIA/CardioMM k-t wrapper constructors technically accept their
other-family keyword fields, but only the fields shown above are consumed by
that family. Production YAML must not put radial-only options on non-radial
paths. `radial/cardiomm` is the static fixed-training implementation and does
not expose its internal 137.5-degree/0.6 constants as constructor parameters;
`kt_radial/cardiomm` is dynamic and does expose the four listed radial fields.

For `random/sdum` and `equispaced/sdum`, let
`L=round(Ky*center_fraction)`. The centered `L` lines are inside the total
nominal sampling budget. Random requires `L<=Ky/R`; equispaced requires the
strict normal-domain condition `L<Ky/R`, plus a finite positive adjusted
spacing. Thus `Ky=156,R8,center_fraction=0.04` (approximately six center
lines) is valid, while ACS20/R8 is not source-faithful for either path.

VISTA source resolution occurs after subject geometry is known. For
`precomputed`, precedence remains `sampling.parameters.mask_path`, top-level
`sampling.mask_root`, then `/path/to/cmrbench-assets/masks`. The exact file is
`mask_VISTA_<Ky>x<T>_acc<R>_<seed>.txt`; a missing file fails without a nearest
file or native fallback. `cmrbench` instead calls
`generate_vista_mask(p=Ky, t=T, acceleration=R, seed=sampling.seed)` using its
validated defaults. It rejects `mask_root` and `mask_path` rather than silently
ignoring disk configuration.

Both sources produce compact Boolean support in `[Ky,T]` order. The shared
path transposes it to `[T,Ky]`, inserts singleton S/Kx axes, and broadcasts it
to canonical `[S,T,Kx,Ky]`. The native implementation is the CMRBench Python
adaptation of the VISTA algorithm (Ahmad et al., MRM 2015), with executable
reference OSU-CMR/VISTA pinned at
`1ad7ea0130247aca220a3e761fc8a64b770900b2`.

ACS remains post-generation: the native generator has no ACS or `ncalib`
concept. By default `force_acs=false`; with `force_acs=true`, explicit
`ncalib` selects centered compact rows
`[(Ky-ncalib)//2 : (Ky-ncalib)//2+ncalib]` across every T through the shared
sampling-level helper. The union can have different final line counts between
frames because overlap with the base VISTA mask may differ. Actual R is always
remeasured from the final canonical union. Precomputed provenance continues to
record its unchanged base/final hashes, counts, accelerations, and bounds.

External-mask YAML normally supplies `mask_path`; direct Python callers may
instead supply an array. Supported explicit layouts are `S,T,Kx,Ky`, `Ky,T`,
`T,Ky`, `Kx,Ky`, and `T,Kx,Ky`. The input is finite, nonempty, exact binary,
and never thresholded. CardioMM use of `external_mask` additionally requires
result provenance `source_identity` and `nominal_acceleration`; these may be
provided in the constructor's inner `parameters` mapping. Actual acceleration
is always remeasured, never trusted from caller metadata.

## Physical sampling versus model conditioning

`sampling.*` defines the physical mask. Model conditioning is a separate
semantic request and never rewrites that mask.

SDUM `sampling_condition` modes are:

- `auto`: optional `acceleration` and `acquisition`; family is resolved from
  an unambiguous physical method.
- `explicit`: required canonical semantic `family` (`uniform`, `gaussian`, or
  `radial`), with optional `acceleration` and `acquisition`.
- `upstream_raw`: required nonempty `raw_mask_type` and literal
  `allow_unmatched: true`, with optional `acceleration` and `acquisition`.
  This is diagnostic-only and can expose `mask_idx=-1`, `acc_idx=-1`, and the
  released negative-index DC fallback.

SDUM auto maps `uniform -> uniform`, `kt_uniform/equispaced -> uniform`,
`kt_gaussian -> gaussian`, and `kt_radial/radial -> radial`. VISTA and random
are ambiguous. External masks can auto-map only when their declared
`family`/`pattern` resolves to an SDUM semantic.

CardioMM accepts `mode=auto|explicit`; both modes allow optional
`acceleration` and `num_low_frequencies`, while explicit requires
`mask_type=uniform|random|radial`. Auto maps `uniform -> uniform`,
`kt_uniform/equispaced -> uniform`, `random -> random`,
`kt_gaussian -> random`, `radial -> radial`, and `kt_radial -> radial`.
VISTA is ambiguous and requires explicit user-forced semantics. In particular,
the CardioMM semantic string need not equal the physical generator name.
`num_low_frequencies` controls model-side sensitivity extraction; it is not
physical `ncalib` and cannot synthesize missing ACS samples.

`random_cartesian` is not in this current selector contract. Its early
CMRBench-native class and direct unit tests remain only for historical/internal
coverage; the default registry, generic resolver, package-level public API and
current examples do not expose it. Historical configs and output metadata are
left unchanged.

## Upstream implementations

The SDUM implementation is a source-faithful port of
`scripts/mri_data/ktSampling.py` plus released wrapper defaults from
NVIDIA/NV-Raw2Insights-MRI commit
`99be712b2e9fd34bf6a2f709373d311354fbda96`. The audited source SHA-256 is
`1c6a71ffe862a88182c36b2eddc955e854b047cfe6a98e1b0b0e2ef36f278db0`;
license is Apache-2.0 with the MONAI Consortium notice retained.

CardioMM sampling calls the external functions under
`CardioMM_dataundersampling/Python_Reimpl_Version/cardiomm_mask_generator`
and `CardioMM_reconstruction/data/subsample.py` at revision
`abb6f8a4e293426736467c9859a5f5755c38c8aa`. No adapted upstream sampling
algorithm is included. The MRBench wrapper checks source fingerprints, preserves
the explicit Gaussian/Python random seed policies and performs canonical
conversion, ACS validation and provenance recording. Missing external code fails
without substituting an algorithm. See [setup and terms](models.md#cardiomm-sampling).

## Canonical Uniform proof contract

Both sources compute a centered `[Kx,ncalib]` strip padded to `[Kx,Ky]` and set
`mask[:,0::R]=1`. Their start expressions are `(Ky-ncalib)//2` (SDUM) and
`floor(Ky/2)-floor(ncalib/2)` (CardioMM). They are identical except when Ky is
even and `ncalib` is odd; at that boundary CardioMM starts one Ky column to the
right. The 48-case matrix covers `ncalib=8,12,15,16,20,24`, R4/R8/R16/R24,
and geometries 192x156x25 and 31x29x7. Every case records array equality,
both binary SHA-256 values, both nonzero counts, both support-derived actual R
values, and centered-strip ACS geometry. Forty-four cases are exactly equal;
the four even-Ky/odd-`ncalib` cases are intentionally different.

Therefore `uniform` remains canonical by default for positive `ncalib<=Ky`,
integer R, and all parity combinations except even Ky + odd `ncalib`. The
canonical protocol rejects only that ambiguous boundary and tells the caller
to select `implementation: sdum` or `implementation: cardiomm`; both
source-faithful implementations are registered for that purpose. It does not
force either formula into the other.

## `ncalib` semantics

All four generated families default to `ncalib=20` for backward compatibility,
but the generic generator layer does not impose ACS20. A positive user value is
accepted when it fits the geometry: non-radial families require
`ncalib<=Ky`; radial requires `ncalib<=Kx` and `ncalib<=Ky`. Uniform,
ktUniform and ktGaussian union a full-Kx centered Ky strip of width `ncalib`;
ktRadial unions a centered `ncalib x ncalib` square. Requested R remains the
nominal user/model category; changing ACS width changes only the measured
actual R.

The public CardioMM launcher and the public SDUM recipes use 20 as their
reference value, but this alone does not prove that their released checkpoints
were trained only with ACS20. CardioMM retains its legacy compatibility label
pending a checkpoint-history audit. For SDUM, a valid source-native non-20 ACS
is runtime-supported and its checkpoint training exposure is recorded as
unknown, rather than inferred to be OOD. CardioMM now passes the configured
truthful `num_low_frequencies` to its released sensitivity extractor; current
formal configs remain fixed at 20. The source can auto-detect a contiguous 1-D
center for null/0, while 2-D radial needs an explicit square size. It consumes
the actual masked tensor and does not synthesize absent central samples. SDUM's released
categorical interface carries family, nominal R and acquisition, not ACS
width; its ACS/CSM path derives the usable central region from the supplied
mask. Invalid `ncalib` provenance or a mask that lacks its **declared**
strip/square fails before inference. If no physical ACS is declared, missing
center support is recorded and deferred to the model runtime rather than made
a generic sampling failure.

## Real implementation differences

The non-uniform implementations are not aliases. At 192x156x25, ACS20:

| family/R | SDUM sampled points / actual R / SHA-256 | CardioMM sampled points / actual R / SHA-256 |
|---|---|---|
| `kt_uniform` R4 | 259200 / 2.888888888888889 / `7c923847…1f58` | 256128 / 2.923538230884558 / `970996ef…25af` |
| `kt_uniform` R8 | 177792 / 4.211663066955 / `388184fc…7850` | 176256 / 4.248366013071895 / `ed7ef3c0…1d0e` |
| `kt_gaussian` R4 seed1 | 244032 / 3.068450039339 / `9b1e4ff4…793` | 240768 / 3.110047846889952 / `d6874747…ab7` |
| `kt_gaussian` R8 seed1 | 170688 / 4.386951631046 / `9c967ac9…eff` | 170304 / 4.396843291995491 / `fec063ad…8b3d` |

The radial ports deliberately retain different upstream rasterizers and their
tie behavior: SDUM uses SciPy `ndimage.rotate`; CardioMM uses
`skimage.transform.rotate(order=0, mode=constant, preserve_range=True)`.

## Historical Random Cartesian note (not a current selector)

`RandomCartesianSampling` first appears, already in its present fixed-count
`default_rng.choice` form, in CMRBench commit
`18600427341155dc58c15451aa9ec5a56077d3d4`. File history, comments and the
separate fastMRI diagnostic scripts provide no evidence of a fastMRI port.
It is therefore recorded as an independent `cmrbench_native` implementation,
not fastMRI-derived.

The code does not materialize a separate native mask array: its stochastic
primitive is a 1-D selected-index set of length `round(phase_size/R)`. With the
default `phase_encoding_axis=3`, only Ky indices are selected and every Kx point
on each selected line is acquired. A time-varying run draws one independent
line-index set per T; `time_varying=false` reuses one set. The result is shared
over S and written directly into `[S,T,Kx,Ky]`. Axis 2 selects Kx instead. There
is no ACS. This differs from genuinely joint k-t masks (`[Ky,T]`) and radial
masks (`[Kx,Ky,T]`), and from fastMRI `RandomMaskFunc` center-fraction plus
outer-line Bernoulli semantics. This historical implementation is not
selectable through current generic CMRBench sampling.

## Acceleration and provenance

`sampling.acceleration` is always the requested/nominal R used by discrete model
conditioning. `actual_acceleration` is recomputed from the final canonical
binary mask as `mask.size / count_nonzero(mask)` and is never used to derive a
category. Every generated result records method, implementation, upstream
repository/commit/source, seed, ACS and generator parameters, native/canonical
layouts, sampled/total points, binary SHA-256, requested R and actual R.
Both R values and the implementation are persisted in subject JSON,
`results.csv`, `experiment_metadata.json`, effective `config.yaml`, and the
concise console PASS line.

## Remote engineering validation

The following user-reported remote results are compatibility/architecture
smoke tests only, not reconstruction-quality evidence:

- CardioMM checkpoint/runtime + CardioMM-native ktGaussian R8: PASS.
- SDUM Small + SDUM ktGaussian R24: PASS.
- SDUM Small + ktUniform R8: the historical strict pre-inference failure is no
  longer policy; auto conditioning uses the unambiguous `uniform` semantic and
  preserves `kt_uniform` as the physical family.
- Earlier passes retained as evidence: CardioMM generated Uniform8, CardioMM
  frozen external Uniform8, SDUM Small ktRadial16, and sampling compatibility
  contract tests.

VISTA supports both the historical precomputed source and the frozen native
CMRBench generator; model identity does not alter either physical mask path.

The public model guide explains conditioning and compatibility:
[model setup and OOD conditions](models.md).
