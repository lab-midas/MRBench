# MRBench interactive dashboard

MRBench organizes evaluations of foundation models for MRI reconstruction around acquisition conditions, signal representations and imaging domains. This is the lightweight, static website source. It runs without a frontend build system or third-party charting libraries.

The Overview introduces the three evaluation dimensions. Experiment names and Details links open the design view; View results opens a separate Results view. A local selector switches views and retains acquisition filters. Acceleration factor and Sampling pattern currently expose the same audited subject-level table through complementary views. The other seven evaluations have concise setup descriptions and a Coming soon result label. There are no published qualitative images in this version.

This development directory is separate from the cloned publishing repository at `E:\mr_recon\mr_recon\CMRBench-public`. Changes here do not update the public website automatically. The existing Pages workflow publishes only `website/` after running the statistics and architecture checks.

## Preview on your computer

From this development-source directory, run:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory website
```

Open <http://127.0.0.1:8765/>. If that port is already in use, choose another port such as 8875 in both the command and browser URL. Keep that terminal running while reviewing; press Ctrl+C to stop it. Opening `index.html` directly cannot fetch the JSON tables.

If you are serving from a remote Linux machine, run the same Python command there and forward its port from a terminal on your own computer:

```text
ssh -L 8765:127.0.0.1:8765 your-user@your-server
```

Then visit the same local URL on your computer. If local port 8765 is occupied, change the first 8765 in the forwarding command and use that local port in the browser.

Experiment links are shareable hash routes, so GitHub Pages needs no route-rewrite configuration:

- `#/overview`
- `#/experiments/acceleration?view=details` or `?view=results`
- `#/experiments/sampling?view=details` or `?view=results`
- `#/protocol`

Filters are saved in the Results URL and preserved when switching to Details and back. Unpublished experiments have no Results link; manually requesting their Results state returns the Details view. Public branding is MRBench; the repository, directories, deployment path and historical protocol identifiers remain unchanged. Relative asset paths work both at the site root and under `/CMRBench/`.

## What the current results mean

The released table contains 1,755 subject endpoints: 65 in-house cardiac cine subjects, three published model variants, three CardioMM-source k-t sampling patterns and nominal R8/16/24. These counts describe this particular cohort, rather than the benchmark's future scope. Relevant coverage is shown inside each experiment, not as homepage counters.

All numeric records, the cohort audit and the descriptive-statistics implementation are preserved. Evaluation normalizes each complete subject source independently by its magnitude p99.5 without clipping. Frames are averaged within subjects; cohort summaries weight subjects equally. Error bars show sample SD, not confidence intervals. Nominal and realized acceleration remain distinct. Exact subject values and condition summaries are available in tooltips; the full CSV and provenance audit can be downloaded within either quantitative view.

Model colors follow the manuscript: **SDUM blue, CardioMM orange, RAM teal**. Published labels remain SDUM Base, CardioMM and RAM DeepInv. Runtime fields remain in the downloadable data but are not presented as a performance ranking.

The Protocol formulas were checked against the actual Python evaluator and prepared-data reader; see [the source verification note](website/docs/protocol-verification.md). See the website's Protocol view and [full evaluation protocol](website/docs/evaluation_protocol.md) for scientific definitions. Existing method documents have not been used to invent additional cohort results.

## Maintain the website

The main files are:

| File                            | Responsibility                                                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `website/data/experiments.json` | Dimensions, experiment descriptions, setup, availability, module choices, optional image collections and source paths |
| `website/app.mjs`               | Overview table, separate Details/Results views and hash routing                                                       |
| `website/protocol.mjs`          | Verified metric descriptions and native MathML equations                                                              |
| `website/experiments/`          | Condition selection and independent acceleration/sampling comparison definitions                                      |
| `website/charts.mjs`            | Existing SVG distribution and mean ± SD renderers; sampling is categorical                                            |
| `website/models.mjs`            | Shared model colors, ordering and variant labels                                                                      |
| `website/media.mjs`             | Optional condition selection and side-by-side PNG image panels                                                        |
| `website/stats.mjs`             | Preserved descriptive statistics, subject pairing and numeric/provenance validation                                   |
| `website/styles.css`            | Shared desktop and small-screen styling                                                                               |

Adding a planned experiment requires a registry entry, without changing the Overview or routing code. Publishing a different quantitative experiment can register its own renderer and filters. Optional image collections stay absent from the page until images are explicitly listed. Read [the maintenance guide](website/docs/dashboard-maintenance.md) before adding results or assets.

## Validate before publishing

Node.js 22 or later is sufficient:

```powershell
node --test website/stats.test.mjs
```

The original five statistics tests are retained. Additional imported tests cover the registry, URLs, filters, model colors, optional media and the frozen data/aggregate snapshot. The existing GitHub Pages workflow runs this same command, so no deployment configuration change is needed. `_qa/` contains ignored local verification artifacts and must not be copied to the publishing repository.

## Synchronize after local review

Run the following yourself in Windows PowerShell after reviewing the dashboard. It copies release files into the existing clone, excludes `.git` and local verification artifacts, and performs no Git operations. It does not delete destination files; obsolete files, if any, should be reviewed separately.

```powershell
$source = (Resolve-Path -LiteralPath 'E:\mr_recon\mr_recon\FoundationModel\public_release\CMRBench_dashboard').Path
$target = (Resolve-Path -LiteralPath 'E:\mr_recon\mr_recon\CMRBench-public').Path
if ($source -eq $target) { throw 'Source and target must differ.' }
if (-not (Test-Path -LiteralPath (Join-Path $target '.git'))) {
    throw 'Target must be your existing cloned publishing repository.'
}
robocopy $source $target /E /XD .git _qa node_modules __pycache__ /XF .env *.log
if ($LASTEXITCODE -ge 8) { throw "Copy failed: robocopy exit code $LASTEXITCODE" }
```

Run `node --test website/stats.test.mjs` from the publishing clone and inspect its changes before deciding to commit and push. The workflow remains configured to deploy `website/` from `main` to <https://midas-tum.github.io/CMRBench/>.

Authors, institutional attribution and the first-party code license remain to be finalized.
