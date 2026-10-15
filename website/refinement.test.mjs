import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { experimentURL, parseRoute, resolveExperimentView, hasResults } from "./registry.mjs";
import { PROTOCOL } from "./protocol.mjs";
const registry = JSON.parse(readFileSync(new URL("./data/experiments.json", import.meta.url)));

test("Details and Results have independent shareable routes; old experiment URLs open Details", () => {
  for (const experiment of registry.experiments) {
    const details = parseRoute(experimentURL(experiment.id, "details"));
    const results = parseRoute(experimentURL(experiment.id, "results"));
    assert.equal(details.id, results.id);
    assert.equal(details.view, "details");
    assert.equal(results.view, "results");
    assert.notEqual(
      experimentURL(experiment.id, "details"),
      experimentURL(experiment.id, "results"),
    );
    assert.equal(parseRoute(`#/experiments/${experiment.id}`).view, "details");
    for (const view of ["details", "results"])
      assert.equal(
        new URL(experimentURL(experiment.id, view), "https://midas-tum.github.io/CMRBench/")
          .pathname,
        "/CMRBench/",
      );
  }
  assert.equal(parseRoute("#/experiments/acceleration?view=unknown").page, "not-found");
});
test("unpublished experiments cannot resolve a fabricated Results page", () => {
  for (const experiment of registry.experiments) {
    assert.equal(resolveExperimentView(experiment, "details"), "details");
    assert.equal(
      resolveExperimentView(experiment, "results"),
      hasResults(experiment) ? "results" : "details",
    );
  }
});
test("switching views preserves the selected acquisition and metric in the URL", () => {
  const filters = new URLSearchParams({
    view: "results",
    sampling: "kt_radial|cardiomm",
    R: "24",
    metric: "SSIM",
  });
  const details = parseRoute(experimentURL("sampling", "details", filters));
  const results = parseRoute(experimentURL(details.id, "results", details.params));
  assert.equal(details.view, "details");
  assert.equal(results.view, "results");
  for (const key of ["sampling", "R", "metric"])
    assert.equal(results.params.get(key), filters.get(key));
});
test("public entry branding changes without changing the Pages path or protocol identifiers", () => {
  const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
  assert.match(
    html,
    /<title>MRBench: Benchmarking Foundation Models for MRI Reconstruction<\/title>/,
  );
  assert.match(html, /aria-label="MRBench overview"/);
  assert.doesNotMatch(html, /CMRBench|<footer/);
  const protocolIds = new Set(Object.values(registry.data_sources).map((s) => s.metric_protocol));
  assert.deepEqual([...protocolIds], ["cmrbench_recon_metrics_v2_subject_p995"]);
});
test("the public scientific reference includes accessible equations without development-version labels", () => {
  assert.deepEqual(
    PROTOCOL.map((section) => section.id),
    ["normalization", "nrmse", "psnr", "ssim"],
  );
  for (const section of PROTOCOL)
    for (const equation of section.equations) {
      assert.ok(equation.label.length > 20);
      assert.ok(equation.math || equation.parts?.length);
    }
  const prose = PROTOCOL.flatMap((section) => section.paragraphs).join(" ");
  assert.doesNotMatch(prose, /Metrics v[12]|CMRBench/);
  assert.match(prose, /99.5th percentile/);
  assert.match(prose, /uniform 7 × 7/);
  assert.match(prose, /data range L = 1/);
});
