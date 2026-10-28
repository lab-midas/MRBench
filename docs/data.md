# Preparing data for CMRBench

The canonical interface describes complete subjects, while permitting different
spatial sizes and coil counts between subjects. Complex k-space is stored as
`[slice,time,kx,ky,coil]`. A physical mask has the same first four dimensions and
contains no coil dimension. A reference is optional for reconstruction and
required for reference-based evaluation.

## The prepared cine reader

For a manifest subject `ExampleCase`, prepare:

```text
preprocess/
  norm_img_ExampleCase.txt.npy       # complex [S,T,X,Y,1]
  cc_smap_15_ExampleCase.txt.npy     # complex [S,1,X,Y,15]
```

Use a CSV with at least `filename` and `split`:

```csv
filename,split,cohort_type
ExampleCase,evaluation,example
```

The filename's stem identifies the subject; split names are explicit selections,
not automatically generated partitions. Additional columns are retained as
metadata. Set `sensitivity_map_coils` to match the filename and array. The reader
forms `coil_images = reference * sensitivity_maps`, then applies the common
centered orthonormal FFT. This establishes a retrospective benchmark source;
it does not recover raw scanner measurements or prescribe preprocessing for a
new cohort. Do not silently replace the reference with RSS.

Model-specific normalization, padding, coil conversion and inverse transforms
belong inside adapters. Evaluation normalization belongs in the evaluator and
does not change saved scientific arrays.

## Raw fastMRI

Install the optional preparation dependencies with `pip install -e '.[fastmri]'`
in the core preparation/evaluation environment. The reader loads raw HDF5
`kspace` and `reconstruction_rss`, performs a centered complex image-domain crop
to reference geometry, and maps static images into the canonical time dimension
`T=1`. It does not use RSS as a complex model input or impose cine temporal
semantics on static data.

Use [`../examples/fastmri_ram.yaml`](../examples/fastmri_ram.yaml) as a template.
`dataset.manifest` is the selected HDF5 basename for this reader, while
`dataset.selection.slices` is `all` or an integer. This differs from cine's CSV
manifest. Choose valid static QC phase index `[0]` and leave cine GIF output off.

Sensitivity maps for RAM can be provided or estimated with the SigPy ESPIRiT
backend **after** applying the actual acquisition mask. Calibration uses only
acquired ACS; unacquired source samples cannot enter map estimation. See
[the detailed fastMRI input guide](fastmri_multicoil_input.md). A new experiment
must state its reference definition; cine's constructed complex reference and
fastMRI's RSS reference should not be pooled into one leaderboard.

## Already acquired undersampled data

The interfaces support `sampling.mode: acquired`: the reader returns the actual
undersampled measurements and their acquisition mask, which preparation
preserves. The shipped cine reader constructs complete starting data, and the
fastMRI path targets full train/validation sources. There is no universal raw
scanner/DICOM loader in this release. Add a reader for your acquired format
rather than discarding its mask or applying a second retrospective mask.
