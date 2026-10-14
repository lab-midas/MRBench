// Subject-level descriptive statistics. No frame pooling or hypothesis tests.
export const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
export function quantile(xs, p) {
  const a = [...xs].sort((x, y) => x - y);
  if (!a.length) return NaN;
  const index = (a.length - 1) * p;
  const low = Math.floor(index);
  return a[low] + (a[Math.min(low + 1, a.length - 1)] - a[low]) * (index - low);
}
export function describe(xs) {
  if (!xs.length) throw new Error("No observations");
  const avg = mean(xs);
  return {
    n: xs.length,
    mean: avg,
    sd: xs.length > 1 ? Math.sqrt(xs.reduce((a, x) => a + (x - avg) ** 2, 0) / (xs.length - 1)) : 0,
    min: Math.min(...xs),
    max: Math.max(...xs),
    median: quantile(xs, 0.5),
    q1: quantile(xs, 0.25),
    q3: quantile(xs, 0.75),
  };
}
export function summarize(rows) {
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row.model)) groups.set(row.model, []);
    groups.get(row.model).push(row);
  }
  return [...groups].map(([model, group]) => ({
    model,
    variant: group[0].variant_or_implementation,
    n: group.length,
    ...Object.fromEntries(
      ["NRMSE", "PSNR", "SSIM", "runtime_sec", "actual_R"].map((metric) => [
        metric,
        describe(group.map((r) => r[metric])),
      ]),
    ),
  }));
}
export function pairRows(rows, a, b) {
  const aa = new Map(rows.filter((r) => r.model === a).map((r) => [r.subject, r]));
  const bb = new Map(rows.filter((r) => r.model === b).map((r) => [r.subject, r]));
  return [...aa]
    .filter(([id]) => bb.has(id))
    .map(([id, row]) => ({ subject: id, a: row, b: bb.get(id) }));
}
export function validateRows(rows) {
  const variants = new Map();
  if (!Array.isArray(rows) || !rows.length) throw new Error("Empty or invalid metrics table");
  const keys = new Set(),
    physical = new Map();
  const identities = new Set();
  for (const row of rows) {
    for (const k of [
      "subject",
      "model",
      "sampling",
      "sampling_implementation",
      "dataset",
      "experiment_id",
      "metric_protocol",
      "binary_mask_sha256",
    ])
      if (typeof row[k] !== "string" || !row[k]) throw new Error(`Missing ${k}`);
    if (row.status !== "passed")
      throw new Error(
        "Metrics file contains failed or missing records; publish their coverage separately",
      );
    for (const k of ["NRMSE", "PSNR", "SSIM", "actual_R", "requested_R", "runtime_sec"])
      if (typeof row[k] !== "number" || !Number.isFinite(row[k])) throw new Error(`Invalid ${k}`);
    const condition = [
      row.dataset,
      row.experiment_id,
      row.metric_protocol,
      row.sampling,
      row.sampling_implementation,
      row.requested_R,
      row.ncalib ?? null,
      row.seed ?? null,
    ];
    const variant = row.variant_or_implementation;
    if (variants.has(row.model) && variants.get(row.model) !== variant) {
      throw new Error("Different model variants require distinct model names or experiment tables");
    }
    variants.set(row.model, variant);
    const key = JSON.stringify([...condition, row.subject, row.model]);
    if (keys.has(key)) throw new Error("Duplicate subject/model/condition");
    keys.add(key);
    identities.add(
      JSON.stringify([
        row.dataset,
        row.experiment_id,
        row.metric_protocol,
        row.ncalib ?? null,
        row.seed ?? null,
      ]),
    );
    const pk = JSON.stringify([...condition, row.subject]);
    if (physical.has(pk)) {
      const anchor = physical.get(pk);
      if (
        anchor.binary_mask_sha256 !== row.binary_mask_sha256 ||
        Math.abs(anchor.actual_R - row.actual_R) > 1e-12
      )
        throw new Error("Cross-model acquisition mismatch");
    } else physical.set(pk, row);
  }
  if (identities.size !== 1)
    throw new Error(
      "Separate datasets, protocols or acquisition levels into distinct experiment tables",
    );
  return rows;
}
