import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
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

test("the shared header contains only the unchanged home logo and secure GitHub link", () => {
  const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
  const header = html.match(/<header class="masthead">([\s\S]*?)<\/header>/)[1];
  assert.equal((header.match(/<a\b/g) || []).length, 2);
  assert.match(header, /class="brand" href="#\/overview" aria-label="MRBench overview"/);
  assert.match(
    header,
    /<span class="brand-mark" aria-hidden="true">M<\/span><span>MRBench<\/span>/,
  );
  const github = header.match(/<a\b[^>]*class="github-link"[\s\S]*?<\/a\s*>/)[0];
  assert.match(github, /href="https:\/\/github.com\/lab-midas\/MRBench"/);
  assert.match(github, /target="_blank"/);
  assert.match(github, /rel="noopener noreferrer"/);
  assert.match(github, />GitHub ↗<\/a\s*>/);
  assert.doesNotMatch(header, /<nav\b|data-page=|>Overview<|>Protocol</);
});

test("Protocol presentation retains canonical scientific definitions and the existing metrics route", () => {
  // Captured from the scientific definitions before the navigation-only refinement.
  assert.equal(
    createHash("sha256").update(JSON.stringify(PROTOCOL)).digest("hex"),
    "5daf435a63a70b62fa0a41f347aae78423f3ed9dd4f2487c5ed0a77be21fd7a6",
  );
  const source = readFileSync(new URL("./protocol.mjs", import.meta.url), "utf8");
  assert.match(source, /"Evaluation Metrics & Protocol"/);
  assert.match(source, /section\("Quantitative Metrics", "metric-definitions", metrics\)/);
  assert.match(
    source,
    /section\("Statistical Aggregation", "statistical-aggregation", aggregation\)/,
  );
  assert.match(source, /section\("Evaluation Protocol", "evaluation-protocol", context\)/);
  assert.deepEqual(parseRoute("#/protocol"), { page: "protocol" });
  assert.deepEqual(parseRoute("#/protocol?section=metrics"), {
    page: "protocol",
    section: "metrics",
  });
});
