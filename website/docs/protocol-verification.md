# MRBench protocol verification

This maintenance note records the source review behind the public Protocol page. It is not linked from the public navigation. No reconstruction metrics or source arrays were recalculated or changed for this refinement.

## Source of truth

Paths below are relative to the original FoundationModel project, not the dashboard-only repository:

| Source                                                                                 | What was verified                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cmrbench/data/inhouse_cine.py`: `load_inhouse_arrays`, `build_inhouse_cine_sample`    | Loads the existing prepared complex image and sensitivity products. The canonical reference is `norm_imgc[..., 0]`; complete benchmark k-space is `fft2c(norm_imgc * sensitivity_maps)`. The reader does not replace the reference by RSS or perform a new scientific metric normalization.                                                                                 |
| `cmrbench/evaluation/reconstruction_metrics.py`: `normalize_subject_magnitude_p995`    | Takes `abs(volume)` in float64, computes `np.percentile(magnitude, 99.5)` over the complete `[S,T,X,Y]` source and divides by that one scalar. Each source is independent. No clipping, gain fitting or epsilon repair. Nonfinite inputs and nonpositive scales raise errors.                                                                                               |
| Same file: `compute_normalized_frame_metrics_v2`                                       | NRMSE is L2 error / L2 normalized reference. MSE is the spatial mean squared difference. PSNR is `10*log10(1/MSE)`, with infinity for exact zero MSE. SSIM is `skimage.metrics.structural_similarity` with `data_range=1`, `win_size=7`, `gaussian_weights=False`. Shape, finite-value, nonnegative magnitude, nonzero reference and spatial-size requirements are checked. |
| Same file: `compute_cine_frame_metrics`, `summarize_subject_metrics`                   | Normalizes each complete source before looping through every slice/phase frame. The subject endpoint is the arithmetic mean of all frame metrics with equal frame weights.                                                                                                                                                                                                  |
| `cmrbench/evaluate.py`: `evaluate_reconstruction`                                      | The generic evaluation entry point explicitly selects `METRIC_PROTOCOL_V2`, passes canonical reference/candidate arrays to the shared evaluator, records sampling provenance and marks exceptions as failed before re-raising.                                                                                                                                              |
| `scripts/slurm/cmrbench_patient_sweeps/aggregate_sweep_metrics.py`: `_validate_record` | Passed records must have finite endpoints, valid mask hashes and matching sampling provenance. Missing/invalid/failed records are distinguished rather than filled with zero values.                                                                                                                                                                                        |
| `public_release/CMRBench/tools/export_metrics.py`: `export`                            | The public export preserves numeric values, requires passed finite records, complete subject groups and cross-model physical-mask agreement. It does not calculate reconstruction metrics.                                                                                                                                                                                  |
| Dashboard `website/stats.mjs`: `describe`, `summarize`, `validateRows`                 | Summarizes already computed subject endpoints with equal subject weight and sample SD using denominator `n−1`. Validates the shipped table; it does not transform images or calculate frame metrics.                                                                                                                                                                        |

The historical protocol identifier `cmrbench_recon_metrics_v2_subject_p995` is unchanged in the data and validation code. Public prose describes the method without development-version labels.

## SSIM expression and dependency defaults

The project explicitly fixes the SSIM window, weighting and range. It leaves K₁, K₂ and covariance normalization at scikit-image defaults. Those defaults were checked against the [official scikit-image implementation](https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/metrics/_structural_similarity.py), consistent with the project's `scikit-image>=0.21` requirement: K₁ = 0.01, K₂ = 0.03 and sample covariance. The returned frame score averages valid local windows after removing the window-radius border.

The public page presents the standard factored local SSIM expression and its spatial average. It does not claim Gaussian weighting or the population-covariance variant. Reviewing dependency defaults is not a recovery of the original server's installed package version; that version is not recorded in the shipped subject-endpoint table. Source-result provenance should be retained for exact environment reproduction.

## Aggregation distinction

Python's `_descriptive_statistics` additionally records population SD (`ddof=0`) across frames as a within-subject diagnostic. The browser plots use sample SD across subject means (`n−1`). These describe different distributions. The public Protocol explicitly identifies the plotted SD as between-subject variation, not a confidence interval.

## Failure and edge-case interpretation

The image evaluator can return infinite PSNR for an exact zero-error frame, but the audited publication path requires finite passed endpoints. The page therefore distinguishes the mathematical edge case from the finite current cohort table. Invalid scales, nonfinite inputs, zero-reference frames and undersized SSIM frames are not silently repaired. No failures or missing rows are synthesized as successful metrics.

## Presentation checks

Formulas are native MathML with accessible verbal labels, not raw Markdown or a remote math-rendering dependency. Their markup is static explanatory content. Numeric data, the statistical implementation and plotting functions remain untouched. The root README documents preview; the maintenance guide documents separate Details/Results routes and extension points.
