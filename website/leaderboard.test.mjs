import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  validateLeaderboard,
  rankLeaderboard,
  formatLeaderboardMetric,
  coverageMessage,
} from "./leaderboard.mjs";
import { describe } from "./stats.mjs";

const data = JSON.parse(readFileSync(new URL("./data/leaderboard.json", import.meta.url)));
const rows = JSON.parse(readFileSync(new URL("./data/main_metrics.json", import.meta.url)));

test("leaderboard exports nine actual conditions and agrees with all 81 audited metric means", () => {
  validateLeaderboard(data);
  assert.equal(data.datasets.length, 1);
  assert.equal(data.datasets[0].label, "In-house cardiac cine (SAX)");
  assert.equal(data.datasets[0].patient_count, 65);
  assert.equal(data.conditions.length, 9);
  for (const condition of data.conditions) {
    assert.equal(condition.measurement_match, "verified");
    for (const model of condition.models) {
      const group = rows.filter(
        (r) =>
          r.model === model.model &&
          r.sampling === condition.sampling &&
          r.requested_R === condition.acceleration,
      );
      assert.equal(group.length, 65);
      for (const metric of data.metrics) {
        assert.ok(
          Math.abs(model.metrics[metric.id].mean - describe(group.map((r) => r[metric.id])).mean) <
            1e-12,
        );
        assert.equal(model.metrics[metric.id].n_patients, 65);
      }
    }
  }
  const serialized = JSON.stringify(data);
  assert.doesNotMatch(serialized, /Case-\d+|"subject"|"patients"\s*:|recovery_source|[A-Z]:[\\/]/);
});

test("every condition ranks SSIM/PSNR descending and NRMSE ascending without rounding first", () => {
  for (const condition of data.conditions)
    for (const metric of data.metrics) {
      const ranked = rankLeaderboard(condition.models, metric);
      for (let i = 1; i < ranked.length; i++) {
        const prior = ranked[i - 1].metrics[metric.id].mean,
          next = ranked[i].metrics[metric.id].mean;
        assert.ok(metric.direction === "higher" ? prior >= next : prior <= next);
      }
      assert.deepEqual(
        ranked.map((r) => r.rank),
        [1, 2, 3],
      );
    }
  const metric = data.metrics[0];
  const a = { model: "A", metrics: { SSIM: { mean: 0.90001, n_patients: 1 } } },
    b = { model: "B", metrics: { SSIM: { mean: 0.90002, n_patients: 1 } } };
  assert.equal(
    formatLeaderboardMetric(a.metrics.SSIM, metric),
    formatLeaderboardMetric(b.metrics.SSIM, metric),
  );
  assert.deepEqual(
    rankLeaderboard([a, b], metric).map((r) => r.model),
    ["B", "A"],
  );
});

test("ties share ranks; missing/nonfinite metrics have no rank; legitimate zeros stay valid", () => {
  const metric = data.metrics.find((m) => m.id === "NRMSE");
  const candidate = (model, mean, n = 1) => ({
    model,
    metrics: { NRMSE: { mean, n_patients: n } },
  });
  const ranked = rankLeaderboard(
    [
      candidate("absent", null, 0),
      candidate("zero", 0),
      candidate("tie", 0),
      candidate("bad", NaN),
      candidate("next", 0.1),
    ],
    metric,
  );
  assert.deepEqual(
    ranked.map((r) => r.rank),
    [1, 1, 3, null, null],
  );
  assert.equal(formatLeaderboardMetric(candidate("a", null, 0).metrics.NRMSE, metric), "N/A");
  assert.equal(formatLeaderboardMetric(candidate("a", 0).metrics.NRMSE, metric), "0.0000");
});

test("coverage warnings distinguish incomplete or different patient subsets from failures", () => {
  const condition = structuredClone(data.conditions[0]);
  assert.equal(coverageMessage(condition, data.datasets[0], data.metrics).warning, false);
  condition.coverage.SSIM = { same_patients: false, common_patients: 60, union_patients: 65 };
  const different = coverageMessage(condition, data.datasets[0], data.metrics);
  assert.equal(different.warning, true);
  assert.match(different.text, /subsets differ/);
  assert.doesNotMatch(different.text, /failure|failed/i);
  condition.coverage.SSIM = { same_patients: true, common_patients: 60, union_patients: 60 };
  assert.match(
    coverageMessage(condition, data.datasets[0], data.metrics).text,
    /Incomplete metric coverage/,
  );
});

test("summary validation rejects contradictory counts, invalid means and mixed metric directions", () => {
  const mutate = (fn) => {
    const copy = structuredClone(data);
    fn(copy);
    return copy;
  };
  assert.throws(
    () => validateLeaderboard(mutate((d) => (d.conditions[0].models[0].metrics.SSIM.mean = null))),
    /Invalid/,
  );
  assert.throws(
    () =>
      validateLeaderboard(mutate((d) => (d.conditions[0].models[0].metrics.SSIM.n_patients = 66))),
    /Invalid/,
  );
  assert.throws(
    () => validateLeaderboard(mutate((d) => (d.metrics[0].direction = "lower"))),
    /Invalid/,
  );
  assert.throws(
    () => validateLeaderboard(mutate((d) => d.conditions.push(d.conditions[0]))),
    /Invalid/,
  );
});

