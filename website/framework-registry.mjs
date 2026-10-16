import { overviewURL, resourceURL, protocolURL } from "./registry.mjs";

// Rectangles are [x, y, width, height] in the authoritative PNG's native pixels.
// The image itself is copied byte-for-byte, never re-exported from the PPTX.
export const FRAMEWORK = Object.freeze({
  image: {
    src: "assets/framework/Fig_1_Workflow.png",
    width: 3681,
    height: 1263,
    sha256: "67abe10f2b2b6fd6731f99c790b34e2c30fa37ce7b4e6f9216a9b25f7da9d48a",
    alt: "The original approved framework figure: data sources, controlled experiment designs, standardized model benchmarking and reconstruction evaluation.",
  },
  hotspots: [
    {
      id: "data-inhouse",
      label: "In-house cine: dataset details",
      rect: [36, 385, 518, 243],
      target: { type: "dataset", id: "inhouse-cine" },
      enabled: true,
    },
    {
      id: "data-cmrxrecon",
      label: "CMRxRecon: dataset details",
      rect: [36, 672, 518, 243],
      target: { type: "dataset", id: "cmrxrecon" },
      enabled: true,
    },
    {
      id: "data-additional",
      label: "Additional data including fastMRI: dataset details",
      rect: [36, 959, 518, 244],
      target: { type: "dataset", id: "additional-data" },
      enabled: true,
    },
    {
      id: "category-acquisition",
      label: "Acquisition conditions: acquisition and measurement experiments",
      rect: [662, 354, 800, 299],
      target: { type: "category", id: "acquisition" },
      enabled: true,
    },
    {
      id: "category-representation",
      label: "Signal fidelity: signal representation and encoding experiments",
      rect: [662, 674, 800, 297],
      target: { type: "category", id: "representation" },
      enabled: true,
    },
    {
      id: "category-domain",
      label: "Domain generalization experiments",
      rect: [662, 990, 800, 243],
      target: { type: "category", id: "domain" },
      enabled: true,
    },
    {
      id: "model-ram",
      label: "RAM: model details",
      rect: [2034, 540, 635, 134],
      target: { type: "model", id: "ram" },
      enabled: true,
    },
    {
      id: "model-sdum",
      label: "SDUM: model details",
      rect: [2034, 697, 635, 135],
      target: { type: "model", id: "sdum" },
      enabled: true,
    },
    {
      id: "model-cardiomm",
      label: "CardioMM: model details",
      rect: [2034, 856, 635, 203],
      target: { type: "model", id: "cardiomm" },
      enabled: true,
    },
    {
      id: "evaluation-metrics",
      label: "Quantitative metrics: Protocol definitions",
      rect: [3337, 308, 306, 407],
      target: { type: "protocol", id: "metrics" },
      enabled: true,
    },
  ],
});

export function hotspotURL(hotspot) {
  const { type, id } = hotspot.target;
  if (type === "category") return overviewURL(id);
  if (type === "protocol") return protocolURL(id);
  return resourceURL(type, id);
}
export function hotspotStyle(rect, image = FRAMEWORK.image) {
  const [x, y, width, height] = rect;
  return `left:${(x / image.width) * 100}%;top:${(y / image.height) * 100}%;width:${(width / image.width) * 100}%;height:${(height / image.height) * 100}%`;
}
export function validateFramework(framework, resources, experiments) {
  const { width, height } = framework.image;
  if (!(width > 0 && height > 0) || !/^assets\//.test(framework.image.src))
    throw new Error("Invalid framework image");
  const ids = new Set();
  for (const hotspot of framework.hotspots) {
    if (
      !hotspot.id ||
      ids.has(hotspot.id) ||
      !hotspot.label ||
      typeof hotspot.enabled !== "boolean"
    )
      throw new Error("Invalid or duplicate framework hotspot");
    ids.add(hotspot.id);
    if (
      !Array.isArray(hotspot.rect) ||
      hotspot.rect.length !== 4 ||
      !hotspot.rect.every(Number.isFinite)
    )
      throw new Error("Invalid hotspot coordinates");
    const [x, y, w, h] = hotspot.rect;
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > width || y + h > height)
      throw new Error("Hotspot exceeds original image");
    const { type, id } = hotspot.target;
    const valid =
      type === "dataset"
        ? resources.datasets.some((e) => e.id === id)
        : type === "model"
          ? resources.models.some((e) => e.id === id)
          : type === "category"
            ? experiments.dimensions.some((e) => e.id === id)
            : type === "protocol" && id === "metrics";
    if (!valid) throw new Error(`Unknown framework destination: ${hotspot.id}`);
  }
  return framework;
}
