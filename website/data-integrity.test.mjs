import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { describe } from "./stats.mjs";
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
test("published numeric files and audit retain their original bytes", () => {
  const frozen = {
    "data/main_metrics.json": "2a5fc99a1dfe7cc6e215a46ac08da08a023e7d8b6eb6ddc9769da165df0ed56a",
    "data/main_metrics.csv": "72f156713907c8062f87e431abcab84470d9aed33dbbd51ee8c7f10fb75e2830",
    "data/main_audit.json": "65f74ce313f01bd1d33efb14d89faee77f1337b2fc5cf953eb1ab2efb6ce49e5",
  };
  for (const [path, expected] of Object.entries(frozen))
    assert.equal(sha256(readFileSync(new URL(path, import.meta.url))), expected, path);
});
test("all 27 condition aggregates equal the pre-refactor snapshot", () => {
  const rows = JSON.parse(readFileSync(new URL("./data/main_metrics.json", import.meta.url)));
  const grouped = {};
  for (const r of rows) {
    const key = [r.model, r.sampling, r.sampling_implementation, r.requested_R].join("|");
    (grouped[key] ??= []).push(r);
  }
  const aggregates = Object.fromEntries(
    Object.entries(grouped).map(([key, group]) => [
      key,
      Object.fromEntries(
        ["NRMSE", "PSNR", "SSIM", "actual_R"].map((metric) => [
          metric,
          describe(group.map((r) => r[metric])),
        ]),
      ),
    ]),
  );
  assert.equal(
    sha256(JSON.stringify(aggregates)),
    "e9822b496b37173cea0898ab5416e72f67d745bd14ff0ad272f05241c2e3dd7d",
  );
});
