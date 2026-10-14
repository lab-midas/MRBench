# Third-party provenance and license status

The first-party CMRBench license and authorship are pending. No license is
implicitly assigned to upstream code, model weights, dependencies or datasets.

| Component | Source | Terms recorded by the existing project audit |
| --- | --- | --- |
| NVIDIA/MONAI sampling adaptation | NV-Raw2Insights-MRI `99be712b2e9fd34bf6a2f709373d311354fbda96`, `scripts/mri_data/ktSampling.py` and wrapper semantics | Apache-2.0; MONAI Consortium attribution retained in `cmrbench/sampling/nvidia.py` |
| VISTA Python port | OSU-CMR/VISTA `1ad7ea0130247aca220a3e761fc8a64b770900b2` | Educational, research and not-for-profit permission with attribution and notice; retained verbatim in `cmrbench/vista/NOTICE.md` |
| CardioMM sampling adaptation | CardioMM_MMCMR-427K `abb6f8a4e293426736467c9859a5f5755c38c8aa` | Academic-use statement reported by the pinned source audit; no standard license file found in that audited commit |
| External reconstruction models | RAM/DeepInv, NVIDIA SDUM, CardioMM and CLIP | External assets not bundled; follow each upstream repository and checkpoint's terms |

The Apache-2.0 text is in `licenses/Apache-2.0.txt`; the original copyright and
attribution statements remain in the adapted source. The VISTA notice is also
linked from the main README. These are component notices, not a proposed common
license for the whole repo.

**Publication item:** obtain explicit redistribution clarification for the
pinned CardioMM sampling adaptation, or replace it with a runtime import from
user-acquired upstream assets before uploading that component. Its current
presence in this local review copy must not be interpreted as a legal grant.
This item matters because the main study uses that physical sampling source;
silently substituting another implementation would change the experiment.

This document summarizes the project's pinned source audit. It is not an
independent confirmation that the current upstream repositories have identical
terms. Source pages were not retrievable during this local preparation; verify
the pinned source and retain any additional required notices before publishing.
