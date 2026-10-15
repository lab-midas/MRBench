# MRBench dashboard source

This static website presents MRI reconstruction robustness experiments. It requires no frontend build step or third-party charting library. Scientific metrics are computed before export; the browser summarizes the unchanged subject endpoints.

Overview groups experiments into three evaluation dimensions. Experiment names and Details links explain the design. View results opens a separate Results view for Acceleration factor or Sampling pattern. The local Details/Results selector preserves filters. Planned experiments have no Results link or placeholder plots.

The Protocol page renders normalization, NRMSE, PSNR and SSIM equations directly, alongside aggregation and comparison principles. Model colors remain SDUM blue, CardioMM orange and RAM teal. No qualitative images are listed in this release.

From the dashboard repository root, run `python -m http.server 8765 --bind 127.0.0.1 --directory website` and open `http://127.0.0.1:8765/`. Run `node --test website/stats.test.mjs` before release. The existing workflow deploys website/ under the unchanged /CMRBench/ project path.

See the root README and docs/dashboard-maintenance.md for file responsibilities and extension points. Internal source verification is documented in docs/protocol-verification.md.
