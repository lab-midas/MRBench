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
    "7af7b00a266ac54cbfb97ded1c01c217454fcbe45993e86f2a5261ee9854cfd2",
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

test("resource registry preserves dataset IDs and independently attributed model sources", () => {
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
  assert.equal(resources.datasets[1].description, "CMRxRecon is a public cardiac MRI dataset.");
  assert.equal(resources.datasets[1].context, undefined);
  assert.deepEqual(resources.datasets[1].links, [
    { label: "Official CMRxRecon project", url: "https://github.com/CmrxRecon" },
    { label: "Dataset publication", url: "https://doi.org/10.1038/s41597-024-03525-4" },
  ]);
  const additional = resources.datasets[2];
  assert.deepEqual(
    additional.sections.map((section) => section.title),
    ["User-provided data", "Other public datasets"],
  );
  assert.match(additional.sections[0].facts[0].text, /\[S, T, Kx, Ky, C\]/);
  assert.equal(additional.sections[0].facts[0].label, "K-space data");
  assert.doesNotMatch(additional.sections[0].facts[0].text, /required/i);
  assert.match(additional.sections[0].facts[1].text, /Binary mask: `\[S, T, Kx, Ky\]`/);
  assert.match(
    additional.sections[0].facts[2].text,
    /Complex-valued or magnitude image: `\[S, T, X, Y\]`/,
  );
  assert.match(additional.sections[0].note, /T = 1/);
  assert.deepEqual(
    additional.sections[0].facts.map((fact) => fact.status),
    ["Required", "Conditional", "Optional", "Model-dependent"],
  );
  assert.deepEqual(
    additional.sections[0].references.map((link) => link.url),
    ["#/experiments/sampling?view=details", "https://github.com/lab-midas/MRBench"],
  );
  assert.equal(
    additional.sections[1].description,
    "MRBench also supports public MRI datasets such as fastMRI, including brain and knee imaging data.",
  );
  assert.equal(additional.sections[1].note, undefined);
  assert.deepEqual(additional.links, [
    { label: "Official fastMRI dataset", url: "https://fastmri.med.nyu.edu/" },
    { label: "Official fastMRI code", url: "https://github.com/facebookresearch/fastMRI" },
  ]);
});

test("in-house details retain the overview identity and the supplied animated cine and parameters", () => {
  const entry = resources.datasets.find((item) => item.id === "inhouse-cine");
  assert.equal(entry.name, "In-house cine");
  assert.equal(entry.page_title, "In-house cardiac cine MRI (short-axis)");
  assert.match(entry.context, /internal.*not publicly available/);
  assert.equal(entry.description, undefined);
  assert.equal(entry.details.preview.caption, "Example short-axis cine.");
  for (const item of entry.details.preparation.slice(0, 2)) {
    assert.equal(item.link.url, "https://users.ece.utexas.edu/~jtamir/files/papers/2486.pdf");
  }
  const gif = readFileSync(new URL(entry.details.preview.src, import.meta.url));
  assert.match(gif.toString("ascii", 0, 6), /^GIF8[79]a$/);
  assert.deepEqual([gif.readUInt16LE(6), gif.readUInt16LE(8)], [640, 480]);
  assert.equal(
    createHash("sha256").update(gif).digest("hex"),
    "862433ffc15a5cb6d01c303b3e96a97c59af12ce945a644ef88f150e9d8fa2bc",
  );
  const facts = Object.fromEntries(entry.details.facts.map((row) => [row.label, row.text]));
  assert.equal(facts.Acquisition, undefined);
  assert.equal(facts["TE / TR"], "1.06 / 2.12 ms");
  assert.equal(facts["Temporal resolution"], "40 ms");
  assert.equal(facts["Cardiac phases"], "25 per cine series");
  assert.equal(entry.details.facts.filter((row) => row.label === "Slice thickness").length, 1);
  const invalid = structuredClone(resources);
  invalid.datasets[0].details.preview.src = "private/source.gif";
  assert.throws(() => validateResources(invalid), /Invalid dataset preview/);
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
