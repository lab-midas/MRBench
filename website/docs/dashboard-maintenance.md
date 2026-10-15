# Maintaining the experiment dashboard

The MRBench website is static HTML, CSS and browser JavaScript. Public branding does not rename the CMRBench repository, existing directories, metric identifiers or deployment path. Its experiment registry drives the Overview and detail pages, so the navigation does not need to be edited whenever an evaluation is added. The current data and `stats.mjs` are a frozen scientific baseline; changing the layout does not require changing either.

## Add an experiment

Add an entry to `data/experiments.json` with a stable, lowercase, hyphenated ID. Choose an existing dimension, or add a dimension with an existing `blue`, `yellow` or `green` visual tone. The entry contains:

- `id`, `name`, `dimension`, and an optional `group` such as Motion;
- a short `description`, an optional `objective`, and a `setup` array of `{label, text}` paragraphs;
- `status`: `planned` until results are explicitly published;
- `quantitative`, `qualitative`, `illustrations`, and `sources`: initially empty arrays.

An empty results list produces a Planned row and a concise forthcoming message. It does not inherit another experiment's numeric table, generate empty charts or create an image placeholder. Illustrations can accompany a planned evaluation without implying that model results are available.

When results are ready, set `status` to `published` and explicitly register at least one quantitative or qualitative collection. The registry validator rejects inconsistent availability and unknown source IDs. The present nine-experiment release is also checked by the deployment tests; update that coverage assertion deliberately when the public scope changes.

## Separate Details and Results

Every experiment has a Details route, for example `#/experiments/acceleration?view=details`. Only published experiments expose `?view=results`. Names and Details links open design explanations and optional illustrations; Results contains only quantitative and optional qualitative results. The title-level selector switches directly between the two views, preserving filter parameters. A manually requested unpublished Results state is canonicalized to Details. Empty image collections remain absent. Existing bare experiment routes still open Details.

The Protocol view renders native MathML from `protocol.mjs`, with no external math library or raw Markdown link. Scientific source verification is recorded in [protocol-verification.md](protocol-verification.md). Keep public method descriptions free of development-version labels while preserving identifiers in the data and validation code.

## Add quantitative results

Each `data_sources` entry points to its own metric JSON, downloadable CSV and audit JSON, and names its metric protocol, cohort and comparison context. The two current experiment entries explicitly share `main-v1`; they are complementary views of the same measurements. A new dataset, acquisition perturbation or metric protocol must not silently reuse this table.

The current JSON schema and scientific validation remain in `stats.mjs`. Each row identifies the subject, dataset, experiment, model/variant, physical sampling implementation, nominal and actual R, metric protocol, mask hash, status and numeric endpoints. The validator checks finite values, unique subject/model/condition keys, model variants and cross-model physical-mask agreement. Different datasets or protocols are kept in separate tables.

For a new plot type, add an independent experiment module and register it in `QUANTITATIVE_MODULES` in `experiments/quantitative.mjs`. The registry chooses that module with `{ "type": "module-name", "source": "source-id" }`. The existing acceleration and sampling modules use the shared filter/distribution shell, each supplying its own comparison chart. A fundamentally different experiment can instead use a separate rendering shell; it need not inherit acceleration filters. The current shell mounts one quantitative comparison module per experiment view.

`experiments/conditions.mjs` owns the current acquisition filters. `charts.mjs` uses the unchanged `describe` implementation to summarize subject endpoints. Sampling patterns are categorical: its grouped means and SD bars must not be joined by trend lines. Use `models.mjs` for every model-associated color and label.

New protocols need explicit protocol-specific labels and validation before publication; the existing shell intentionally accepts only the published Metrics v2 protocol. Preserve nominal versus actual R, subject matching and the physical-mask/model-conditioning distinction.

## Add illustrations or qualitative comparisons

Place approved, de-identified, pre-rendered PNGs under:

```text
assets/experiments/<experiment-id>/<condition-id>/
```

Do not add raw multi-coil k-space, model caches or other large scientific arrays for browser display. Verify frame selection, intensity windows, error-map semantics and mask provenance before publishing images. Motion examples must reflect the actual simulator, including phase-line replacement or line-dependent k-space ramps as appropriate.

Both `illustrations` and `qualitative` accept collections with the same schema. This is an illustrative schema only; the paths below are not published assets:

```json
{
  "id": "comparison",
  "title": "Reconstruction comparison",
  "caption": "Describe frame selection, display scaling and source provenance here.",
  "conditions": [
    {
      "id": "condition-id",
      "label": "The verified acquisition or perturbation condition",
      "panels": [
        {
          "role": "reference",
          "label": "Reference",
          "src": "assets/experiments/experiment-id/condition-id/reference.png",
          "alt": "A useful description of this reference frame"
        },
        {
          "role": "reconstruction",
          "model": "SDUM",
          "variant": "base",
          "label": "Reconstruction",
          "src": "assets/experiments/experiment-id/condition-id/sdum.png",
          "alt": "A useful description of this reconstruction"
        }
      ]
    }
  ]
}
```

The renderer provides a condition selector when several conditions exist and lays out that condition's panels side by side. Supported roles are `reference`, `reconstruction`, `error`, `mask`, `input` and `zero-filled`. Error maps and additional model panels use the same list. Model borders use the shared manuscript colors; optional variant metadata preserves model labels. Empty collections render nothing. No qualitative examples are listed in this release.

## Check and release

Run `node --test website/stats.test.mjs`. The original five statistical tests remain intact and import the architecture and frozen-data checks. File hashes protect the current CSV, JSON and audit; an aggregate snapshot checks every existing group for NRMSE, PSNR, SSIM and actual R. Update a frozen-data expectation only as part of a deliberate, separately audited scientific-data release.

Serve `website/` over HTTP and review Overview, both quantitative pages, Protocol, and a planned experiment. Check filters, keyboard tooltips, matching subject highlights, direct filter URLs and browser Back. At small widths, only the plots scroll horizontally; the page should not overflow. Check relative links under the GitHub Pages `/CMRBench/` prefix. The current workflow deploys `website/` unchanged and needs no router fallback.

Use the PowerShell synchronization recipe in the root README only after local review. It excludes `.git` and `_qa/`; it does not commit, push or delete files from the destination clone. The independent publishing repository is never an automatic build output.
