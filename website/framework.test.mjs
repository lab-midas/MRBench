import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { FRAMEWORK, hotspotURL, hotspotStyle, validateFramework } from "./framework-registry.mjs";
import { validateResources } from "./resources.mjs";
import { overviewURL, resourceURL, protocolURL, parseRoute, routeAnchorId } from "./registry.mjs";

const read = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const resources = read("./data/resources.json");
const experiments = read("./data/experiments.json");

test("framework retains the approved PNG bytes, dimensions and exactly ten scoped hotspots", () => {
  const png = readFileSync(new URL(FRAMEWORK.image.src, import.meta.url));
  assert.equal(
    createHash("sha256").update(png).digest("hex"),
    "67abe10f2b2b6fd6731f99c790b34e2c30fa37ce7b4e6f9216a9b25f7da9d48a",
  );
  assert.equal(FRAMEWORK.image.sha256, createHash("sha256").update(png).digest("hex"));
  assert.deepEqual(
    [png.readUInt32BE(16), png.readUInt32BE(20)],
    [FRAMEWORK.image.width, FRAMEWORK.image.height],
  );
  assert.equal(FRAMEWORK.hotspots.length, 10);
  assert.ok(FRAMEWORK.hotspots.every((item) => item.enabled));
  assert.deepEqual(
    FRAMEWORK.hotspots.reduce(
      (counts, { target }) => ({ ...counts, [target.type]: (counts[target.type] || 0) + 1 }),
      {},
    ),
    { dataset: 3, category: 3, model: 3, protocol: 1 },
  );
  validateFramework(FRAMEWORK, resources, experiments);
});

test("all hotspot rectangles are disjoint, relative and map to registered destinations", () => {
  for (const item of FRAMEWORK.hotspots) {
    const [x, y, w, h] = item.rect;
    const style = hotspotStyle(item.rect);
    assert.equal((style.match(/%/g) || []).length, 4);
    assert.ok(!style.includes("px"));
    const values = [...style.matchAll(/:([\d.]+)%/g)].map((match) => Number(match[1]));
    for (const [index, expected] of [
      (x / 3681) * 100,
      (y / 1263) * 100,
      (w / 3681) * 100,
      (h / 1263) * 100,
    ].entries())
      assert.ok(Math.abs(values[index] - expected) < 1e-10);
    const route = parseRoute(hotspotURL(item));
    if (item.target.type === "category")
      assert.equal(routeAnchorId(route), `category-${item.target.id}`);
    else if (item.target.type === "protocol")
      assert.equal(routeAnchorId(route), "metric-definitions");
    else assert.equal(route.id, item.target.id);
    for (const other of FRAMEWORK.hotspots.filter((hotspot) => hotspot.id !== item.id)) {
      const [a, b, c, d] = other.rect;
      assert.ok(
        x + w <= a || a + c <= x || y + h <= b || b + d <= y,
        `${item.id} overlaps ${other.id}`,
      );
    }
  }
});

test("new shareable hash routes keep the MRBench Pages prefix and stable anchors", () => {
  const base = "https://lab-midas.github.io/MRBench/";
  const routes = [
    overviewURL("acquisition"),
    overviewURL("representation"),
    overviewURL("domain"),
    protocolURL("metrics"),
    ...resources.datasets.map((e) => resourceURL("dataset", e.id)),
    ...resources.models.map((e) => resourceURL("model", e.id)),
  ];
  for (const route of routes) {
    const url = new URL(route, base);
    assert.equal(url.pathname, "/MRBench/");
    assert.notEqual(parseRoute(url.hash).page, "not-found");
  }
  assert.deepEqual(parseRoute(overviewURL("domain")), { page: "overview", category: "domain" });
  assert.deepEqual(parseRoute(protocolURL("metrics")), { page: "protocol", section: "metrics" });
  assert.equal(routeAnchorId(parseRoute("#/overview")), null);
  assert.equal(parseRoute("#/protocol?section=unregistered").page, "not-found");
  assert.equal(parseRoute("#/datasets/cmrxrecon/extra").page, "not-found");
  assert.equal(
    new URL(FRAMEWORK.image.src, base).pathname,
    "/MRBench/assets/framework/Fig_1_Workflow.png",
  );
});

test("resource registry has three minimal datasets and independently attributed model sources", () => {
  validateResources(resources);
  assert.deepEqual(
    resources.datasets.map((e) => e.id),
    ["inhouse-cine", "cmrxrecon", "additional-data"],
  );
  assert.deepEqual(
    resources.models.map((e) => e.id),
    ["ram", "sdum", "cardiomm"],
  );
  const papers = ["2503.08915", "2512.17137", "2512.21652"];
  for (const [index, model] of resources.models.entries()) {
    assert.equal(
      model.links.find((link) => link.label === "Original paper").url,
      `https://arxiv.org/abs/${papers[index]}`,
    );
    assert.match(
      model.links.find((link) => link.label === "Original code").url,
      /^https:\/\/github.com\//,
    );
    assert.match(model.attribution, /Original work by/);
  }
  assert.match(resources.datasets[1].context, /planned dataset extension/);
  assert.match(resources.datasets[2].context, /No fastMRI cohort results/);
});

test("registries reject unknown targets, invalid geometry and missing source attribution", () => {
  const invalidTarget = structuredClone(FRAMEWORK);
  invalidTarget.hotspots[0].target.id = "absent";
  assert.throws(
    () => validateFramework(invalidTarget, resources, experiments),
    /Unknown framework destination/,
  );
  const outside = structuredClone(FRAMEWORK);
  outside.hotspots[0].rect[0] = -1;
  assert.throws(() => validateFramework(outside, resources, experiments), /exceeds original image/);
  const missing = structuredClone(resources);
  missing.models[0].links.pop();
  assert.throws(() => validateResources(missing), /Missing model attribution/);
});

test("category destinations stay valid when more experiments join an existing dimension", () => {
  const expanded = structuredClone(experiments);
  expanded.experiments.push({ ...expanded.experiments[0], id: "future-acquisition" });
  assert.equal(validateFramework(FRAMEWORK, resources, expanded), FRAMEWORK);
});
