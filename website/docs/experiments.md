# Robustness experiments

The main study asks how released models respond to acquisition changes. Its
physical mask is shared between models, while each model retains its own internal
normalization, sensitivity estimation and conditioning. The same distinction
applies to the simulation pilots below. Code availability and result availability
are tracked separately in the dashboard.

| Dimension | Included code | Results included in the dashboard |
| --- | --- | --- |
| Acceleration and k-t sampling | Patient-sweep runners and samplers | Full main table: 65 subjects × 3 models × 3 families × R8/16/24 |
| VISTA and ACS dependence | VISTA sampler, ACS/preflight support and QC plotting | Empty placeholder; no result asset or cohort metrics included |
| Complex measurement noise | Noise controller, noise sweeps, fully sampled QC | Empty placeholder; no noise table or figure included |
| Cardiac mistriggering | Motion simulator/controller and QC plotting | Implemented; no local final metrics or prepared motion assets |
| Respiratory motion | Motion simulator/controller and QC plotting | Implemented translation surrogate; no local final cohort results |
| Global input scaling | Scaling controller and QC plotting | Implemented; separate single-slice fastMRI diagnostics exist privately and are not the cardiac cohort |
| Magnitude-derived k-space | Construction controller and QC plotting | Implemented; no locally available final cohort results |
| Dataset shift | Static fastMRI reader and preparation | Input capability; further cross-dataset benchmark results planned |

## Main retrospective study

The released table uses SDUM Base, RAM DeepInv and CardioMM with physical
`implementation: cardiomm`, k-t uniform/Gaussian/pseudo-radial masks,
nominal R=8/16/24, ncalib=20 and seed=1. Pseudo-radial generation retains
`angle4next=137.5`, `cropcorner=true`, `radial_acceleration_scale=0.6`.
These are Cartesian rasterized masks; the framework does not imply a NUFFT
radial acquisition. Actual R is computed from the realized complete mask.

Each source result is a full-subject Metrics v2 endpoint. Frame means are
computed within each subject, then subjects receive equal weight in cohort
summaries. The published final groups contain 65 subjects. Historical patient
task manifests were frozen for 66 subjects and 1,782 tasks, and 27 records from
one now-ineligible subject are excluded by final aggregation. The public table
and cohort audit retain the final scope. Do not regenerate or reinterpret that
historical study from the toy manifest.

## Running the included pilots

The copied controllers keep their original editable CONFIG block. Configure
the subject, manifest, preprocess root, cache directory, model YAML asset paths
and runtime environment variables before running them from the repo root:

```bash
python scripts/noise/run_kspace_noise.py
python scripts/motion/run_motion_reconstruction.py
python scripts/scaling/run_global_scaling.py
python scripts/magnitude_image/run_magnitude_derived_kspace.py
```

The subject is selected from the configured evaluation split. Only the toy
quick start runs without external data and model assets. Controllers launch
model workers in separate interpreters; visualization scripts read completed
caches and do not run inference. The original study cohort/manifests and raw
cache arrays are not included in this upload candidate.

The recipe defaults below describe the copied settings at snapshot time, not
verified settings for every historical animation. Record the resulting metadata
and hashes for any actual run.

## Complex Gaussian measurement noise

`scripts/noise/run_kspace_noise.py` uses k-t uniform, nominal R8, ncalib20,
sampling seed1 and noise seed1, with target SNRs 25/30/35 dB. Noise is circular
complex Gaussian with real/imaginary standard deviation `sigma_complex/sqrt(2)`.
It is added only to acquired multi-coil entries, including acquired ACS. Sigma is
derived from the clean acquired-support signal energy; samples are not rescaled
to force the realized SNR to equal its target.

SNR is `20*log10(||clean acquired data||2 / ||added acquired noise||2)`.
Record both target and realized SNR, complex sigma and noise seed. Each frozen
noisy measurement is shared by all models. External maps/mask remain fixed;
SDUM/CardioMM internal map estimation sees noisy ACS. RAM retains its existing
sigma-zero integration, so this is an external perturbation test, not a new
noise-aware RAM condition.

The existing private R=1 diagnostic uses the clean fully sampled **coil-combined baseline**
as its comparator. It is a measurement-noise sanity example and cannot rank the
three foundation models. That private display normalizes each source by subject p99.5
and uses scale-invariant anatomy error maps; it supplies no cohort metrics and is excluded from this release.

## Cardiac mistriggering

`simulate_mistriggering` randomly replaces a configured fraction of acquired
phase-encoding lines with lines from another cardiac phase, using a fixed seed
and the same corruption across coils. Rounding determines the realized line
count. `preserve_acs` optionally protects explicitly declared acquired ACS;
record that setting, eligible/replaced line counts and seed.

The controller's current `MISTRIGGER_FRACTION=0.00` is a no-corruption setting.
Choose and record nonzero levels for a stress experiment. This is a phase-line
inconsistency surrogate, not a full ECG acquisition model.

## Respiratory translation surrogate

`simulate_respiratory_motion` applies a line-dependent k-space phase ramp for
sinusoidal in-plane translation along canonical X. Defaults in the copied
controller are amplitude 10 pixels, four cycles across Ky, zero respiratory
phase, `MOTION_TYPE=respiratory` and `PRESERVE_ACS=False`. It restores ACS only
when explicitly requested and finally applies the physical mask.

The displacement is indexed by the Ky line order; the code does not model true
scanner timing, nonrigid deformation, through-plane motion or subject breathing
traces. Report it as a controlled surrogate and preserve its exact settings.

## Global input scaling

`scripts/scaling/run_global_scaling.py` freezes a k-t uniform nominal-R8,
ncalib20, seed1 anchor and applies positive factors 0.01/0.1/1/10/100 to the
input measurement. The physical mask, reference and external sensitivity maps
are unchanged. Zero-filled output scales linearly; candidate outputs are kept
in their raw adapter units without dividing by the factor. Each model's internal
normalization/estimation is preserved.

Scale-invariant Metrics v2 and anatomy displays can hide global output-gain
changes. Include raw-amplitude diagnostics and scale ratios alongside structural
quality before interpreting scaling robustness. Separate fastMRI VCC/coil
comparison diagnostics are not interchangeable with this cardiac experiment.

## Magnitude-derived synthetic k-space

`scripts/magnitude_image/run_magnitude_derived_kspace.py` forms
`fft2c(complex_sensitivity_maps * abs(complex_reference))`, then samples it using
the selected physical mask. Defaults are k-t uniform, nominal R8/16/24,
ncalib20 and seed1. This removes the reference object's phase while retaining
complex sensitivity-map phase. It is not a claim that every complex phase in the
coil data is removed. The magnitude reference remains the comparison target;
generated caches are explicitly labeled as synthetic.

## Sampling/ACS and future dataset shifts

VISTA and ACS study assets need explicit physical-mask provenance, effective R,
ACS geometry/size and model conditioning for each condition. Archived GIFs alone
cannot establish paired measurements. Add quantitative tables only after a
versioned audit verifies subject coverage and masks.

For future CMRxRecon/fastMRI cohorts, state dataset selection, train/evaluation
overlap, reference construction, preprocessing, static/cine geometry and protocol
separately. Keep their tables independent from the in-house main study.
