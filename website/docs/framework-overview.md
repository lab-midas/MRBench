# Interactive Framework Overview

The approved framework is shown above the original experiment matrix. It is one PNG, with ten transparent HTML links positioned as percentages of its native dimensions. The four-panel composition, text, colors, sampling examples, icons and MRI images are unchanged. Hover and keyboard focus add only an outline to the selected link. No new result images, numeric results, metric definitions or scientific claims of dataset validation were introduced.

## Figure provenance

The authoritative source is `E:/PhD/ISMRM_27/Figures/Fig_1_Workflow.png`. It was copied without re-encoding to `assets/framework/Fig_1_Workflow.png`.

- Native dimensions: **3681 × 1263**.
- SHA256, source and distributed copy: `67abe10f2b2b6fd6731f99c790b34e2c30fa37ce7b4e6f9216a9b25f7da9d48a`.
- `E:/PhD/paper-CMRBench/framework.pptx` was inspected as a ZIP/XML package for slide text and shape geometry. It was not edited, rendered into a replacement figure or distributed with the website.
- PPT and PNG text differ in places. The PNG is authoritative: its **CMRBench**, **Acquisition robustness** and **Signal robustness** labels remain exactly as supplied. The dashboard's surrounding MRBench branding is unchanged.

The image scales proportionally on desktop and laptop screens. Below its minimum useful display width, only the figure's container scrolls horizontally. It is never cropped, stretched or reorganized into a new layout. A short mobile instruction explains the horizontal scroll.

## Ten enabled interactions

Rectangles are `[x, y, width, height]` in original PNG pixels; rendering converts them to percentages. All destinations are hash routes and retain the `/MRBench/` deployment prefix.

| Original figure region | Rectangle | Actual destination |
| --- | --- | --- |
| In-house cine | `[36,385,518,243]` | `#/datasets/inhouse-cine` |
| CMRxRecon | `[36,672,518,243]` | `#/datasets/cmrxrecon` |
| Additional data | `[36,959,518,244]` | `#/datasets/additional-data` |
| Blue acquisition category | `[662,354,800,299]` | `#/overview?category=acquisition` |
| Orange signal category | `[662,674,800,297]` | `#/overview?category=representation` |
| Purple domain category | `[662,990,800,243]` | `#/overview?category=domain` |
| RAM | `[2034,540,635,134]` | `#/models/ram` |
| SDUM | `[2034,697,635,135]` | `#/models/sdum` |
| CardioMM | `[2034,856,635,203]` | `#/models/cardiomm` |
| Quantitative Metrics | `[3337,308,306,407]` | `#/protocol?section=metrics` |

Category links smoothly scroll to `category-acquisition`, `category-representation` and `category-domain` on the existing table. These IDs come from the experiment dimensions, independently of the number of experiments. Selecting the same category again also scrolls correctly. Refreshing an anchored route restores the destination; browser Back/Forward uses normal hash history. The category heading receives focus without an initial focus-induced jump. No hover action navigates.

The metrics route uses the `metric-definitions` section in `protocol.mjs`, showing the Quantitative Metrics heading and normalization followed by NRMSE, PSNR and SSIM. The page is titled Evaluation Metrics & Protocol and groups its original scientific content into three sections. Its route and equations are preserved. The shared header now contains only the unchanged MRBench home logo and a GitHub external link to the public repository; the framework and matrix provide the main exploration links.

## Architecture and changed files

Added release files:

- `assets/framework/Fig_1_Workflow.png`: exact source copy.
- `framework-registry.mjs`: image provenance and centralized enabled hotspot mapping.
- `framework.mjs`: responsive original-image rendering and accessible link overlay.
- `data/resources.json`: dataset/model metadata, attribution and verified URLs.
- `resources.mjs`: shared minimal dataset/model view and metadata validation.
- `framework.test.mjs`: focused source-image, scope, geometry, route and registry tests.
- `docs/framework-overview.md`: this implementation record and final page content.

Modified release files:

- `app.mjs`: mount the framework above the existing table; extend navigation for resources and anchors; focus and same-category scrolling.
- `registry.mjs`: extend the current hash parser and add URL/anchor helpers, preserving experiment routes and filters.
- `styles.css`: framework overlay, internal horizontal scrolling, focus indication and compact resource-page styling.
- `architecture.test.mjs`: import the additional framework tests into the existing deployment test entry point.
- `docs/dashboard-maintenance.md` and the release-root `README.md`: current public URLs, routing and metadata maintenance instructions.

Local `_qa/framework/` artifacts hold the baseline hashes, PPT inspection, screenshots, browser checks and a `/MRBench/` preview server. They are ignored and are not release files. No deployment workflow or publishing-clone file was edited; no commit or push was executed.

## Sources and scientific verification

Descriptions were checked against the released local documentation and adapters in the sibling development release, including `public_release/CMRBench/docs/models.md`, `docs/data.md`, `cmrbench/models/adapters/` and `cmrbench/data/fastmri.py`. These were read only. The current experiment registry and 1,755-record table determine what is actually published. Upstream model papers describe their authors' models, not new MRBench results.

Verified primary sources, checked 2026-10-15:

| Resource | Verified original sources | Basis for the page |
| --- | --- | --- |
| RAM | [Paper](https://arxiv.org/abs/2503.08915), [author repository](https://github.com/matthieutrs/ram) | Paper author list and non-iterative physics-aware architecture; repository identifies RAM. Local adapter identifies the DeepInv implementation used here. |
| SDUM | [Paper](https://arxiv.org/abs/2512.17137), [official NV-Raw2Insights-MRI repository](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI) | Paper author list, unrolled reconstruction, learned sensitivities and acquisition conditioning; official repository corresponds to SDUM. The upstream repository has moved from the older NVIDIA URL recorded in the local adapter; the informational page uses its verified current location. |
| CardioMM | [Paper](https://arxiv.org/abs/2512.21652), [author repository](https://github.com/wangziblake/CardioMM_MMCMR-427K) | Paper names CardioMM and describes semantic context plus physics-informed data consistency; local adapter supplies the implementation context. |
| CMRxRecon | [Official project organization](https://github.com/CmrxRecon), [dataset publication](https://doi.org/10.1038/s41597-024-03525-4) | Public cardiac MRI k-space dataset/challenge; existence is distinguished from MRBench integration and publication status. |
| fastMRI | [Official initiative/dataset site](https://fastmri.med.nyu.edu/), [official code](https://github.com/facebookresearch/fastMRI) | MRI dataset and research initiative; the selected static multi-coil preparation path is identified from local source, without claiming universal validation. |
| In-house cine | Current experiment registry, data provenance and local prepared-data documentation | No external download link, scanner parameters, acquisition protocol or private participant/institutional information is invented. |

## Validation and local preview

Run `node --test website/stats.test.mjs` from the release root. The existing workflow runs this same entry point. The original 19 tests, six framework tests and two header/Protocol regression tests all **pass (27 total)**. Existing numerical baselines and integrity tests are unchanged.

Byte comparisons against the pre-task SHA256 snapshot cover the three published data files, experiment registry, data-integrity test, statistics implementation/test entry point, charts, model colors, Protocol and Pages workflow. The source image has an independent exact-byte hash assertion. The framework tests cover exactly ten active regions, non-overlap, native dimensions, relative coordinates, resource attribution, MRBench base-path routing, valid destinations and future experiments within a category.

Browser verification and any remaining limitations are recorded in the local `_qa/framework/verification.md` after the final checks. The first release intentionally leaves every other framework area static. There are no additional dataset results or qualitative result galleries.

For ordinary preview, from `E:/mr_recon/mr_recon/FoundationModel/public_release/CMRBench_dashboard/`:

```powershell
python -m http.server 8875 --bind 127.0.0.1 --directory website
```

Open `http://127.0.0.1:8875/`. To reproduce the tested Pages prefix, run `python _qa/framework/serve_preview.py` instead and open `http://127.0.0.1:8875/MRBench/`. Use one server at a time and Ctrl+C to stop it. The task-owned verification server is stopped on completion. For a remote Linux server, run the HTTP command there, forward port 8875 with `ssh -L 8875:127.0.0.1:8875 your-user@your-server` from your computer, then open the local URL.

## Final Dataset and Model page content

The following is a review snapshot of `data/resources.json`, the maintained source of the six pages. Update that metadata to change future page content; the renderer needs no per-resource branches. Each page also has `← Back to overview` and the unchanged top navigation.

### Dataset: In-house cine

The current in-house cardiac cine dataset used in MRBench's published acceleration and sampling experiments. Benchmark measurements are constructed from prepared complex reference images and sensitivity maps.

Only anonymized quantitative endpoints are published for this cohort. Source data and private participant information are not distributed through the dashboard.


### Dataset: CMRxRecon

CMRxRecon provides public cardiac MRI k-space datasets and reconstruction challenges for the research community.

CMRxRecon is a planned dataset extension in MRBench. The results currently published here use the in-house cine cohort; no CMRxRecon cohort results are included.

[Official CMRxRecon project](https://github.com/CmrxRecon) · [Dataset publication](https://doi.org/10.1038/s41597-024-03525-4)

### Dataset: Additional data / fastMRI

MRBench is designed to extend to additional MRI datasets, including fastMRI, an external MRI dataset and research initiative.

The framework includes an input-preparation path for selected static multi-coil brain and knee data. This does not establish validation for every fastMRI format or model combination. No fastMRI cohort results are published in this dashboard.

[Official fastMRI dataset](https://fastmri.med.nyu.edu/) · [Official fastMRI code](https://github.com/facebookresearch/fastMRI)

### Model: RAM

Reconstruct Anything Model

A lightweight, non-iterative reconstruction model that incorporates acquisition physics to address a range of imaging inverse problems, including MRI.

Original work by Matthieu Terris, Samuel Hurault, Maxime Song and Julian Tachella.

The published MRBench cohort uses the DeepInv implementation of RAM. Its MRI adapter receives multi-coil measurements, a physical mask and complex sensitivity maps.

Paper link title: *Reconstruct Anything Model: a lightweight general model for computational imaging*.

[Original paper](https://arxiv.org/abs/2503.08915) · [Original code](https://github.com/matthieutrs/ram)

### Model: SDUM

Scalable Deep Unrolled Model

An MRI reconstruction framework combining cascaded unrolled reconstruction, learned coil sensitivity estimation and acquisition-aware data consistency and conditioning.

Original work by Puyang Wang, Pengfei Guo, Keyi Chai, Jinyuan Zhou, Daguang Xu and Shanshan Jiang.

The published MRBench cohort evaluates SDUM Base. The adapter preserves the released temporal-window inference and internal sensitivity-map estimation. The official implementation is distributed as NV-Raw2Insights-MRI.

Paper link title: *SDUM: A Scalable Deep Unrolled Model for Universal MRI Reconstruction*.

[Original paper](https://arxiv.org/abs/2512.17137) · [Original code](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI)

### Model: CardioMM

Cardiovascular MRI reconstruction foundation model

A generalist model for accelerated cardiovascular MRI reconstruction that combines semantic acquisition context with physics-informed data consistency.

Original work by Zi Wang and colleagues.

MRBench uses a pinned official source implementation with framewise inference, internal sensitivity-map estimation and acquisition-metadata conditioning.

Paper link title: *Enabling Ultra-Fast Cardiovascular Imaging Across Heterogeneous Clinical Environments with A Generalist Foundation Model and Multimodal Database*.

[Original paper](https://arxiv.org/abs/2512.21652) · [Original code](https://github.com/wangziblake/CardioMM_MMCMR-427K)
