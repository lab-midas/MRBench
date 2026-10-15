import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { describe } from "./stats.mjs";
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
// Git checkouts may convert CRLF to LF. Remove only the CR in a CRLF pair;
// preserve every other byte, including lone CRs, BOMs and the final newline.
function canonicalLineEndings(bytes) {
  const canonical = Buffer.allocUnsafe(bytes.length);
  let length = 0;
  for (let index = 0; index < bytes.length; index++) {
    if (bytes[index] === 0x0d && bytes[index + 1] === 0x0a) continue;
    canonical[length++] = bytes[index];
  }
  return canonical.subarray(0, length);
}
test("published numeric files and audit retain their canonical bytes", () => {
  // Frozen baselines after CRLF -> LF only: no trimming or reserialization.
  const frozen = {
    "data/main_metrics.json": "2a5fc99a1dfe7cc6e215a46ac08da08a023e7d8b6eb6ddc9769da165df0ed56a",
    "data/main_metrics.csv": "151a3b11fb029994dcf0fbaac995de93cfaa2b49950d9b892e6257e46c6de439",
    "data/main_audit.json": "65f74ce313f01bd1d33efb14d89faee77f1337b2fc5cf953eb1ab2efb6ce49e5",
  };
  for (const [path, expected] of Object.entries(frozen))
    assert.equal(
      sha256(canonicalLineEndings(readFileSync(new URL(path, import.meta.url)))),
      expected,
      path,
    );
});
test("canonical integrity hashing tolerates only CRLF/LF differences", () => {
  const lf = Buffer.from("model,NRMSE\nSDUM,0.125\n");
  const crlf = Buffer.from("model,NRMSE\r\nSDUM,0.125\r\n");
  const mixed = Buffer.from("model,NRMSE\r\nSDUM,0.125\n");
  const expected = sha256(lf);
  for (const representation of [lf, crlf, mixed]) {
    assert.deepEqual(canonicalLineEndings(representation), lf);
    assert.equal(sha256(canonicalLineEndings(representation)), expected);
  }
  const changes = {
    "numeric value": Buffer.from("model,NRMSE\nSDUM,0.126\n"),
    whitespace: Buffer.from("model,NRMSE\nSDUM, 0.125\n"),
    "missing final newline": Buffer.from("model,NRMSE\nSDUM,0.125"),
    "UTF-8 BOM": Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), lf]),
    "lone CR": Buffer.from("model,NRMSE\rSDUM,0.125\n"),
    "non-text byte": Buffer.concat([lf, Buffer.from([0xff])]),
  };
  for (const [change, bytes] of Object.entries(changes))
    assert.notEqual(sha256(canonicalLineEndings(bytes)), expected, change);
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
