import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { availability, experimentURL, parseRoute, validateRegistry } from "./registry.mjs";
import { MODEL_COLORS, modelLabel } from "./models.mjs";
import { initialSelection, selectCondition, patterns } from "./experiments/conditions.mjs";
import { hasPublishedMedia, validateMedia } from "./media.mjs";
import "./refinement.test.mjs";
const registry = JSON.parse(readFileSync(new URL("./data/experiments.json", import.meta.url)));
const rows = JSON.parse(readFileSync(new URL("./data/main_metrics.json", import.meta.url)));

test("registry separates two published views from seven planned evaluations", () => {
  validateRegistry(registry);
  assert.equal(registry.dimensions.length, 3);
  assert.equal(registry.experiments.length, 9);
  assert.deepEqual(
    registry.experiments.filter((e) => e.status === "published").map((e) => e.id),
    ["acceleration", "sampling"],
  );
  for (const entry of registry.experiments) {
    if (entry.status === "planned") {
      assert.equal(availability(entry), "Planned");
      assert.deepEqual(entry.sources, []);
      assert.deepEqual(entry.quantitative, []);
    }
    assert.equal(hasPublishedMedia(entry.illustrations), false);
    assert.equal(hasPublishedMedia(entry.qualitative), false);
    assert.equal(parseRoute(experimentURL(entry.id)).id, entry.id);
  }
  for (const source of Object.values(registry.data_sources))
    for (const field of ["metrics_path", "csv_path", "audit_path"]) {
      assert.ok(!source[field].startsWith("/"), "Paths must preserve the Pages project prefix");
      assert.ok(existsSync(new URL(source[field], import.meta.url)));
    }
});
test("registry rejects inconsistent availability, duplicates and unlinked sources", () => {
  const mutate = (fn) => {
    const copy = structuredClone(registry);
    fn(copy);
    return copy;
  };
  assert.throws(
    () =>
      validateRegistry(
        mutate((r) => {
          r.experiments[2].status = "published";
        }),
      ),
    /availability/,
  );
  assert.throws(
    () =>
      validateRegistry(
        mutate((r) => {
          r.experiments[1].id = "acceleration";
        }),
      ),
    /duplicate/,
  );
  assert.throws(
    () =>
      validateRegistry(
        mutate((r) => {
          r.experiments[0].sources = [];
        }),
      ),
    /Unlinked/,
  );
});
test("hash routes and result links work beneath the GitHub Pages project path", () => {
  assert.equal(parseRoute("").page, "overview");
  assert.equal(parseRoute("#/overview").page, "overview");
  assert.equal(parseRoute("#/protocol").page, "protocol");
  assert.equal(parseRoute("#/experiments/unknown").page, "experiment");
  assert.equal(parseRoute("#/experiments/acceleration/extra").page, "not-found");
  const route = parseRoute(experimentURL("sampling", true) + "&R=16&metric=SSIM");
  assert.equal(route.id, "sampling");
  assert.equal(route.params.get("view"), "results");
  assert.equal(route.params.get("R"), "16");
  assert.equal(route.params.get("metric"), "SSIM");
  const url = new URL(experimentURL("acceleration", true), "https://midas-tum.github.io/CMRBench/");
  assert.equal(url.pathname, "/CMRBench/");
});
test("filters select existing conditions and restore safe defaults for invalid links", () => {
  const params = new URLSearchParams({ sampling: "kt_gaussian|cardiomm", R: "24", metric: "PSNR" });
  const selected = initialSelection(rows, params);
  const group = selectCondition(rows, selected.sampling, selected.acceleration);
  assert.equal(group.length, 195);
  assert.ok(group.every((r) => r.sampling === "kt_gaussian" && r.requested_R === 24));
  assert.equal(selected.metric, "PSNR");
  assert.equal(patterns(rows).length, 3);
  const fallback = initialSelection(
    rows,
    new URLSearchParams("sampling=vista&R=999&metric=runtime_sec"),
  );
  assert.equal(fallback.metric, "NRMSE");
  assert.equal(fallback.acceleration, 8);
  assert.equal(selectCondition(rows, fallback.sampling, fallback.acceleration).length, 195);
});
test("manuscript model colors and published variant labels are centralized", () => {
  assert.deepEqual(MODEL_COLORS, { SDUM: "#446d9d", CardioMM: "#c77853", RAM: "#208977" });
  assert.equal(modelLabel("SDUM", rows), "SDUM Base");
  assert.equal(modelLabel("RAM", rows), "RAM DeepInv");
  assert.equal(modelLabel("CardioMM", rows), "CardioMM");
});
test("optional media accept condition-specific PNG comparisons and reject raw arrays", () => {
  assert.equal(hasPublishedMedia([]), false);
  const collection = [
    {
      id: "example",
      title: "Comparison",
      conditions: [
        {
          id: "r8",
          label: "Nominal R8",
          panels: [
            {
              role: "reference",
              label: "Reference",
              src: "assets/experiments/noise/reference.png",
              alt: "Description of the reference frame",
            },
          ],
        },
      ],
    },
  ];
  assert.equal(hasPublishedMedia(collection), true);
  validateMedia(collection);
  const raw = structuredClone(collection);
  raw[0].conditions[0].panels[0].src = "private/coil_data.npy";
  assert.throws(() => validateMedia(raw), /PNG/);
});
