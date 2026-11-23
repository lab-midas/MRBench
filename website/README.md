# Dashboard development

Static source for the [MRBench project](../README.md), with no frontend build step.

## Local preview

From the repository root:

```text
python -m http.server 8895 --bind 127.0.0.1 --directory website
```

Open [the local dashboard](http://127.0.0.1:8895/). Keep the server running while reviewing; use Ctrl+C to stop it.

## Validation

With Node.js 22 or later, run from the repository root:

```text
node --test website/stats.test.mjs
```

The existing Pages workflow validates and publishes `website/` under `/MRBench/`.

## Refresh leaderboard results

These commands apply to the development project, which contains `tools/`.

```text
python tools/prepare_leaderboard.py --input /path/to/all_metrics_long.csv
```

This writes `website/data/leaderboard.json`. The source CSV is read only. Finite metrics from passed records are averaged within each patient, then across patients with equal weight, separately for each model, sampling pattern and nominal R. The inspected source already contains one endpoint per patient and condition, following the evaluator's equal-frame subject aggregation. No images are re-evaluated.

The export includes means, per-metric patient counts, cohort-overlap counts and sampling-match status, without patient identifiers or source paths. Missing/non-finite metrics are excluded, never replaced by zero. The table shows `N/A` and no rank when the ranking metric is unavailable. Different patient subsets are explicitly flagged; missing values do not establish a reconstruction failure.

For another dataset, use its own CSV with `--dataset-id`, `--dataset-label` and `--append`. This replaces the named dataset while retaining other exported datasets. Dropdowns use the exported metadata and conditions; no future-dataset placeholders are shipped. Each input must contain one metric protocol, sampling implementation, ACS setting and seed, and one variant per model.

Run the Python export tests separately in the development project:

```text
python -B -m unittest discover -s tools -p test_prepare_leaderboard.py
```

The public website test suite requires neither Python nor `tools/`. The deployed leaderboard reads only static JSON and requires no Python server.

See the [evaluation protocol](docs/evaluation_protocol.md), [scientific source verification](docs/protocol-verification.md) and [maintenance guide](docs/dashboard-maintenance.md) for methods and extension instructions.
