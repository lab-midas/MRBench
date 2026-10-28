# Third-party notices

The MRBench first-party license will be determined in November. No license is
assigned here to upstream code, weights, dependencies or datasets.

- **SDUM/NVIDIA sampling:** `cmrbench/sampling/nvidia.py` adapts
  `scripts/mri_data/ktSampling.py` and wrapper semantics from
  [NV-Raw2Insights-MRI](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI/tree/99be712b2e9fd34bf6a2f709373d311354fbda96),
  revision `99be712b2e9fd34bf6a2f709373d311354fbda96`, under Apache-2.0.
  MONAI Consortium attribution is retained in the modified source; the license
  text is in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
- **VISTA:** the Python port derives from
  [OSU-CMR/VISTA](https://github.com/OSU-CMR/VISTA/tree/1ad7ea0130247aca220a3e761fc8a64b770900b2),
  revision `1ad7ea0130247aca220a3e761fc8a64b770900b2`. Its educational, research
  and not-for-profit redistribution permission, copyright, disclaimers and
  author attribution are retained in [cmrbench/vista/NOTICE.md](cmrbench/vista/NOTICE.md).
- **CardioMM sampling:** algorithms from
  [CardioMM_MMCMR-427K](https://github.com/wangziblake/CardioMM_MMCMR-427K/tree/abb6f8a4e293426736467c9859a5f5755c38c8aa),
  revision `abb6f8a4e293426736467c9859a5f5755c38c8aa`, are external dependencies.
  The upstream academic-use statement does not expressly grant redistribution;
  no adapted upstream algorithms are included. MRBench's sampling module only
  calls user-installed upstream functions and converts/validates their outputs.
  Users must comply with upstream terms; see [setup](docs/models.md#cardiomm-sampling).
- **Reconstruction models:** RAM/DeepInv, NVIDIA SDUM, CardioMM and CLIP source,
  checkpoints and dependency stacks are not bundled. Their respective terms apply
  separately. CardioMM's confidential author-provided checkpoint is excluded.

These are component notices, not a common license for the repository.
