// Shared by quantitative plots, legends and optional image comparisons.
export const MODEL_COLORS = Object.freeze({
  SDUM: "#446d9d",
  CardioMM: "#c77853",
  RAM: "#208977",
});
const MODEL_ORDER = ["SDUM", "CardioMM", "RAM"];
export function modelColor(model) {
  return MODEL_COLORS[model] || "#8267a3";
}
export function modelNames(rows) {
  const rank = (name) =>
    MODEL_ORDER.includes(name) ? MODEL_ORDER.indexOf(name) : MODEL_ORDER.length;
  return [...new Set(rows.map((row) => row.model))].sort(
    (a, b) => rank(a) - rank(b) || a.localeCompare(b),
  );
}
export function modelLabel(model, rows = []) {
  const variant = rows.find((row) => row.model === model)?.variant_or_implementation;
  if (model === "SDUM" && variant) return `SDUM ${variant[0].toUpperCase()}${variant.slice(1)}`;
  if (model === "RAM" && variant) return `RAM ${variant === "deepinv" ? "DeepInv" : variant}`;
  return model;
}
