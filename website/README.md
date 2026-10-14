# The CMRBench results dashboard

This buildless HTML/JavaScript site reads versioned result files. All inference,
metric calculation and scientific normalization happen before export. The browser
calculates descriptive summaries over the shipped subject endpoints.

```bash
python -m http.server 8000 --directory website
```

Open `http://localhost:8000`. Direct `file://` loading cannot fetch the JSON
tables. The site loads no remote fonts, chart libraries, analytics or user data.

## What's included

Results uses the final 1,755-row main study. Choose a sampling implementation,
nominal acceleration and metric. Dot/box plots show subject variation; acceleration
curves show mean ± SD, and the paired plot matches subjects between selected
models. Click a point to highlight the same subject in both plots. Missing or
failed metrics are never converted to zero. No hypothesis tests are added.

Experiments follows the evaluation program. This first release includes only
acceleration and k-t sampling quantitative results. Qualitative evaluation and
all other test results are empty placeholders. Protocol explains the common
evaluator, scientific limitations and provenance.

## Deploy this directory with GitHub Pages

The included `.github/workflows/pages.yml` deploys **only `website/`** through
GitHub Actions. It assumes the public repo's default branch is `main`; edit that
branch if needed. Enable Pages with Source = GitHub Actions in the repo settings.
The workflow can also be run manually. No weights or patient arrays are deployed.

Only the isolated `CMRBench/` folder is an upload candidate. Do not upload its
private parent folder, source linkage maps, local development dependencies or
original experiment arrays. Read the first-party and third-party license items
before public upload. This local preparation does not publish anything.

## Add results without rebuilding the site

Export one dataset/protocol/acquisition level to a new data subdirectory with
`tools/export_metrics.py`, keeping the private ID map outside the public checkout.
Add an entry to `data/experiments.json` with `metrics_path` and `csv_path` relative
to the website root. The quantitative selector derives its options from that
registry. Do not overwrite an existing frozen experiment ID with a different
protocol. Extra noise/ACS/scaling levels need separate entries, or an intentional
extension of the condition controls and pairing keys. See `docs/extending.md`.

Add qualitative assets under `assets/` and entries to `data/cases.json`. Preserve
display and provenance fields. Never imply a Reference/ZF/matched-model comparison
when only a candidate GIF is available. No existing result figure or animation is included in this release.

## Validate

```bash
node --test website/stats.test.mjs
python -m unittest discover -s tests -v
```

The core functions validate finite scores, duplicate condition keys and shared
physical-mask identities. Confirm the UI at desktop and mobile widths, including
filter changes, paired selection, data downloads and missing-result errors.

## Keep documentation current

The deployment workflow renders the English Markdown documentation before
publishing. To regenerate it locally after editing the README or guides:

```bash
python -m pip install -e '.[docs]'
python tools/build_docs.py
```

This renderer copies only public documentation and example configurations into
the static site. It reads no original experiment arrays or private parent files.
