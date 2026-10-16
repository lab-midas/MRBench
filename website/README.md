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

See the [evaluation protocol](docs/evaluation_protocol.md), [scientific source verification](docs/protocol-verification.md) and [maintenance guide](docs/dashboard-maintenance.md) for methods and extension instructions.
