export const patternNames = Object.freeze({
  kt_uniform: "k-t uniform",
  kt_gaussian: "k-t Gaussian",
  kt_radial: "k-t pseudo-radial",
  uniform: "Uniform",
  vista: "VISTA",
  random: "Random",
  equispaced: "Equispaced",
  radial: "Pseudo-radial",
});
export const METRICS = Object.freeze(["NRMSE", "PSNR", "SSIM"]);
const resultPatternNames = Object.freeze({
  kt_uniform: "kt-uniform",
  kt_gaussian: "kt-Gaussian",
  kt_radial: "kt-radial (pseudo)",
});
export const patternKey = (row) => `${row.sampling}|${row.sampling_implementation}`;
export function patternLabel(key) {
  const [name, implementation] = key.split("|");
  return `${patternNames[name] || name} · ${implementation}`;
}
export function resultPatternLabel(key) {
  return resultPatternNames[key.split("|")[0]] ?? patternLabel(key);
}
export function patterns(rows) {
  const order = ["kt_uniform", "kt_gaussian", "kt_radial"];
  const rank = (key) => {
    const n = order.indexOf(key.split("|")[0]);
    return n < 0 ? order.length : n;
  };
  return [...new Set(rows.map(patternKey))].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}
export function accelerations(rows, sampling) {
  return [
    ...new Set(
      rows.filter((r) => !sampling || patternKey(r) === sampling).map((r) => r.requested_R),
    ),
  ].sort((a, b) => a - b);
}
export function selectCondition(rows, sampling, acceleration) {
  return rows.filter((r) => patternKey(r) === sampling && r.requested_R === Number(acceleration));
}
export function initialSelection(rows, params = new URLSearchParams(), defaultAcceleration) {
  const choices = patterns(rows);
  const sampling = choices.includes(params.get("sampling")) ? params.get("sampling") : choices[0];
  const rs = accelerations(rows, sampling);
  const requestedR = Number(params.get("R") ?? defaultAcceleration);
  return {
    sampling,
    acceleration: rs.includes(requestedR) ? requestedR : rs[0],
    metric: METRICS.includes(params.get("metric")) ? params.get("metric") : "NRMSE",
  };
}
