import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, pairRows, validateRows, summarize } from "./stats.mjs";
const rows = JSON.parse(readFileSync(new URL("./data/main_metrics.json", import.meta.url)));
test("published main table is finite, unique, paired and mask-matched", () => {
  validateRows(rows);
  assert.equal(rows.length, 1755);
  assert.equal(new Set(rows.map((r) => r.subject)).size, 65);
  const groups = new Map();
  for (const r of rows) {
    const key = `${r.model}|${r.sampling}|${r.requested_R}`;
    if (!groups.has(key)) groups.set(key, new Set());
    groups.get(key).add(r.subject);
  }
  assert.equal(groups.size, 27);
  for (const group of groups.values()) assert.equal(group.size, 65);
});
test("subject summaries use sample SD and interpolated quantiles", () => {
  const result = describe([1, 2, 3, 4]);
  assert.equal(result.mean, 2.5);
  assert.ok(Math.abs(result.sd - Math.sqrt(5 / 3)) < 1e-12);
  assert.equal(result.median, 2.5);
  assert.equal(result.q1, 1.75);
  assert.equal(result.q3, 3.25);
});
test("pairing joins subject identifiers rather than row order", () => {
  const group = rows.filter((r) => r.sampling === "kt_uniform" && r.requested_R === 8);
  const pairs = pairRows([...group].reverse(), "SDUM", "RAM");
  assert.equal(pairs.length, 65);
  for (const pair of pairs) {
    assert.equal(pair.a.subject, pair.b.subject);
    assert.equal(pair.a.binary_mask_sha256, pair.b.binary_mask_sha256);
  }
  const summary = summarize(group);
  assert.equal(summary.length, 3);
  for (const model of summary) assert.equal(model.n, 65);
});
test("duplicate, invalid and unmatched acquisition records fail visibly", () => {
  assert.throws(() => validateRows([rows[0], rows[0]]), /Duplicate/);
  assert.throws(() => validateRows([{ ...rows[0], NRMSE: NaN }]), /Invalid/);
  assert.throws(() => validateRows([{ ...rows[0], status: "failed" }]), /failed/);
  const other = rows.find(
    (r) =>
      r.subject === rows[0].subject &&
      r.sampling === rows[0].sampling &&
      r.requested_R === rows[0].requested_R &&
      r.model !== rows[0].model,
  );
  assert.throws(
    () => validateRows([rows[0], { ...other, binary_mask_sha256: "different" }]),
    /mismatch/,
  );
});
test("mixed datasets/protocols cannot enter one leaderboard", () => {
  assert.throws(
    () => validateRows([rows[0], { ...rows[1], variant_or_implementation: "another_variant" }]),
    /Different model variants/,
  );
  assert.throws(
    () => validateRows([rows[0], { ...rows[1], dataset: "another_dataset" }]),
    /Separate/,
  );
  assert.throws(() => validateRows([rows[0], { ...rows[1], ncalib: 24 }]), /Separate/);
});

// These architecture and integrity checks also run in the existing Pages workflow.
import "./architecture.test.mjs";
import "./data-integrity.test.mjs";
