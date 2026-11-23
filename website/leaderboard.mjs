import { el, fetchJSON } from "./dom.mjs";
import { modelColor, modelLabel } from "./models.mjs";

export function validateLeaderboard(data) {
  const fail = () => {
    throw new Error("Invalid leaderboard summary");
  };
  const count = (value) => Number.isInteger(value) && value >= 0;
  if (data.schema_version !== 1 || !data.datasets?.length || !data.conditions?.length) fail();
  const directions = { SSIM: "higher", PSNR: "higher", NRMSE: "lower" };
  if (data.metrics?.length !== 3 || new Set(data.metrics.map((m) => m.id)).size !== 3) fail();
  for (const metric of data.metrics)
    if (
      directions[metric.id] !== metric.direction ||
      !count(metric.decimals) ||
      metric.decimals > 8
    )
      fail();
  const datasets = new Map();
  for (const dataset of data.datasets) {
    if (
      !dataset.id ||
      !dataset.label ||
      datasets.has(dataset.id) ||
      !count(dataset.patient_count) ||
      !dataset.models?.length ||
      !dataset.patterns?.length
    )
      fail();
    if (
      new Set(dataset.models.map((m) => m.id)).size !== dataset.models.length ||
      new Set(dataset.patterns.map((p) => p.id)).size !== dataset.patterns.length
    )
      fail();
    for (const model of dataset.models) if (!model.id || !model.variant) fail();
    for (const pattern of dataset.patterns) if (!pattern.id || !pattern.label) fail();
    datasets.set(dataset.id, dataset);
  }
  const seen = new Set();
  for (const condition of data.conditions) {
    const dataset = datasets.get(condition.dataset);
    if (
      !dataset ||
      !dataset.patterns.some((p) => p.id === condition.sampling) ||
      !Number.isFinite(condition.acceleration) ||
      condition.acceleration <= 0
    )
      fail();
    const key = JSON.stringify([condition.dataset, condition.sampling, condition.acceleration]);
    if (seen.has(key)) fail();
    seen.add(key);
    if (
      !Array.isArray(condition.models) ||
      condition.models.length !== dataset.models.length ||
      new Set(condition.models.map((m) => m.model)).size !== dataset.models.length
    )
      fail();
    for (const row of condition.models) {
      if (
        !dataset.models.some((m) => m.id === row.model) ||
        !count(row.patients_observed) ||
        row.patients_observed > dataset.patient_count ||
        !count(row.excluded_observations)
      )
        fail();
      for (const metric of data.metrics) {
        const value = row.metrics?.[metric.id];
        if (!value || !count(value.n_patients) || value.n_patients > row.patients_observed) fail();
        if (value.n_patients === 0 ? value.mean !== null : !Number.isFinite(value.mean)) fail();
      }
    }
    for (const metric of data.metrics) {
      const coverage = condition.coverage?.[metric.id];
      if (
        !coverage ||
        typeof coverage.same_patients !== "boolean" ||
        !count(coverage.common_patients) ||
        !count(coverage.union_patients) ||
        coverage.common_patients > coverage.union_patients ||
        coverage.union_patients > dataset.patient_count
      )
        fail();
      const ns = condition.models.map((row) => row.metrics[metric.id].n_patients);
      if (
        coverage.common_patients > Math.min(...ns) ||
        coverage.union_patients < Math.max(...ns) ||
        (coverage.same_patients &&
          ns.some((n) => n !== coverage.common_patients || n !== coverage.union_patients))
      )
        fail();
    }
    if (!["verified", "unverified", "mismatch"].includes(condition.measurement_match)) fail();
  }
  for (const dataset of data.datasets)
    for (const pattern of dataset.patterns)
      if (!data.conditions.some((c) => c.dataset === dataset.id && c.sampling === pattern.id))
        fail();
  return data;
}

export function rankLeaderboard(rows, metric) {
  const score = (row) =>
    row.metrics[metric.id].n_patients > 0 && Number.isFinite(row.metrics[metric.id].mean)
      ? row.metrics[metric.id].mean
      : null;
  const direction = metric.direction === "lower" ? 1 : -1;
  const available = rows
    .filter((row) => score(row) !== null)
    .sort((a, b) => direction * (score(a) - score(b)) || a.model.localeCompare(b.model));
  const unavailable = rows
    .filter((row) => score(row) === null)
    .sort((a, b) => a.model.localeCompare(b.model));
  let previous, rank;
  return [
    ...available.map((row, index) => {
      const value = score(row);
      if (index === 0 || value !== previous) rank = index + 1;
      previous = value;
      return { ...row, rank };
    }),
    ...unavailable.map((row) => ({ ...row, rank: null })),
  ];
}

export function formatLeaderboardMetric(value, metric) {
  return value?.n_patients > 0 && Number.isFinite(value.mean)
    ? value.mean.toFixed(metric.decimals)
    : "N/A";
}

export function coverageMessage(condition, dataset, metrics) {
  const coverage = metrics.map((m) => condition.coverage[m.id]);
  const differing = metrics.filter((m) => !condition.coverage[m.id].same_patients);
  let text,
    warning = false;
  if (differing.length) {
    warning = true;
    text = `Patient subsets differ across models for ${differing.map((m) => m.id).join(", ")}. Means use each model's available patients, not an identical-case cohort. Shared patients: ${metrics.map((m) => `${m.id} ${condition.coverage[m.id].common_patients}`).join("; ")}.`;
  } else if (coverage.every((c) => c.common_patients === dataset.patient_count)) {
    text = `${dataset.patient_count} matched patients across all models for every metric. Each patient has equal weight.`;
  } else {
    warning = true;
    text = `Incomplete metric coverage. Shared patients: ${metrics.map((m) => `${m.id} ${condition.coverage[m.id].common_patients}`).join("; ")} of ${dataset.patient_count}. Missing values are excluded, not replaced with zero.`;
  }
  if (condition.measurement_match !== "verified") {
    warning = true;
    text +=
      condition.measurement_match === "mismatch"
        ? " Physical sampling measurements differ across records."
        : " Physical sampling matching could not be verified.";
  }
  return { text, warning };
}

let summary;
function loadSummary() {
  if (!summary)
    summary = fetchJSON("data/leaderboard.json")
      .then(validateLeaderboard)
      .catch((error) => {
        summary = undefined;
        throw error;
      });
  return summary;
}

export async function mountLeaderboard(root) {
  const section = el("section", null, {
    class: "leaderboard",
    "aria-labelledby": "leaderboard-heading",
  });
  const header = el("header", null, { class: "leaderboard-heading" });
  const icon = el("span", null, { class: "leaderboard-icon", "aria-hidden": "true" });
  icon.append(el("span"), el("span"), el("span"));
  const title = el("div");
  title.append(
    el("h2", "Reconstruction leaderboard", { id: "leaderboard-heading" }),
    el("p", "Compare reconstruction fidelity under matched evaluation settings."),
  );
  header.append(icon, title);
  const body = el("div");
  body.append(el("p", "Loading reconstruction results…", { role: "status" }));
  section.append(header, body);
  root.append(section);
  try {
    const data = await loadSummary();
    body.replaceChildren();
    renderControls(body, data);
    section.dataset.state = "ready";
  } catch {
    body.replaceChildren(
      el(
        "p",
        "The reconstruction leaderboard is unavailable. Please reload the page or try again later.",
        { class: "error", role: "alert" },
      ),
    );
    section.dataset.state = "error";
  }
}

function renderControls(root, data) {
  const controls = el("div", null, { class: "leaderboard-controls" });
  function control(id, label, help) {
    const field = el("div");
    const heading = el("label", label, { for: id });
    heading.append(
      el("span", " ⓘ", { class: "leaderboard-info", title: help, "aria-hidden": "true" }),
    );
    const select = el("select", null, { id, title: help, "aria-describedby": `${id}-help` });
    field.append(heading, select, el("span", help, { id: `${id}-help`, class: "sr-only" }));
    controls.append(field);
    return select;
  }
  const datasetSelect = control(
    "leaderboard-dataset",
    "Dataset",
    "Only datasets with exported results are listed.",
  );
  const patternSelect = control(
    "leaderboard-sampling",
    "Sampling pattern",
    "Physical sampling pattern for the selected dataset. In-house kt-radial masks use pseudo-radial support on a Cartesian grid.",
  );
  const accelerationSelect = control(
    "leaderboard-acceleration",
    "Acceleration factor",
    "Nominal acceleration setting. Actual acceleration includes ACS and can differ from nominal R.",
  );
  const metricSelect = control(
    "leaderboard-rank",
    "Rank by",
    "Higher SSIM or PSNR and lower NRMSE rank first. Rankings use full-precision patient means; exact ties share a rank.",
  );
  const options = (select, items, previous) => {
    select.replaceChildren(
      ...items.map((item) => el("option", item.label, { value: String(item.id) })),
    );
    if (items.some((item) => String(item.id) === previous)) select.value = previous;
  };
  options(datasetSelect, data.datasets);
  options(
    metricSelect,
    data.metrics.map((m) => ({
      id: m.id,
      label: `${m.id} ${m.direction === "higher" ? "↑" : "↓"}`,
    })),
  );
  const scroll = el("div", null, {
    class: "leaderboard-table-wrap",
    role: "region",
    "aria-label": "Reconstruction rankings",
    tabindex: "0",
  });
  const table = el("table", null, { class: "leaderboard-table" });
  scroll.append(table);
  const status = el("p", null, {
    class: "leaderboard-note",
    role: "status",
    "aria-live": "polite",
  });
  root.append(
    controls,
    scroll,
    status,
    el("p", "On small screens, scroll the table horizontally to compare all metrics.", {
      class: "leaderboard-mobile-note",
    }),
  );
  function dataset() {
    return data.datasets.find((d) => d.id === datasetSelect.value);
  }
  function updatePatterns() {
    options(patternSelect, dataset().patterns, patternSelect.value);
    updateAccelerations();
  }
  function updateAccelerations() {
    const rs = [
      ...new Set(
        data.conditions
          .filter((c) => c.dataset === datasetSelect.value && c.sampling === patternSelect.value)
          .map((c) => c.acceleration),
      ),
    ].sort((a, b) => a - b);
    options(
      accelerationSelect,
      rs.map((r) => ({ id: r, label: `R = ${r}` })),
      accelerationSelect.value,
    );
    updateTable();
  }
  function updateTable() {
    const condition = data.conditions.find(
      (c) =>
        c.dataset === datasetSelect.value &&
        c.sampling === patternSelect.value &&
        c.acceleration === Number(accelerationSelect.value),
    );
    const metric = data.metrics.find((m) => m.id === metricSelect.value);
    table.replaceChildren();
    table.append(
      el(
        "caption",
        `${dataset().label}, ${patternSelect.selectedOptions[0].label}, nominal R = ${condition.acceleration}. Patient-weighted means, ranked by ${metric.id}.`,
        { class: "sr-only" },
      ),
    );
    const head = el("thead"),
      heading = el("tr");
    heading.append(el("th", "Rank", { scope: "col" }), el("th", "Model", { scope: "col" }));
    for (const definition of data.metrics) {
      const attrs = {
        scope: "col",
        "data-metric": definition.id,
        title: definition.unit ? `${definition.id} (${definition.unit})` : definition.id,
      };
      if (definition.id === metric.id) {
        attrs.class = "ranking-metric";
        attrs["aria-sort"] = metric.direction === "higher" ? "descending" : "ascending";
      }
      heading.append(
        el("th", `${definition.id} ${definition.direction === "higher" ? "↑" : "↓"}`, attrs),
      );
    }
    head.append(heading);
    table.append(head);
    const body = el("tbody");
    for (const result of rankLeaderboard(condition.models, metric)) {
      const row = el("tr", null, { "data-model": result.model });
      row.append(
        el(
          "td",
          result.rank ?? "—",
          result.rank === null ? { "aria-label": "Not ranked: selected metric unavailable" } : {},
        ),
      );
      const model = dataset().models.find((m) => m.id === result.model);
      const label = modelLabel(result.model, [
        { model: result.model, variant_or_implementation: model.variant },
      ]);
      const name = el("th", null, { scope: "row" });
      name.append(
        el("span", label, {
          style: `color:${modelColor(result.model)}`,
          title: `Evaluated implementation: ${model.variant}`,
        }),
      );
      row.append(name);
      for (const definition of data.metrics) {
        const value = result.metrics[definition.id];
        const cell = el("td", null, {
          "data-metric": definition.id,
          ...(definition.id === metric.id ? { class: "ranking-metric" } : {}),
        });
        cell.append(
          el("span", formatLeaderboardMetric(value, definition), {
            class: "metric-value",
            ...(value.mean === null ? {} : { title: String(value.mean) }),
          }),
          el("small", `n = ${value.n_patients}`, { class: "metric-count" }),
        );
        row.append(cell);
      }
      body.append(row);
    }
    table.append(body);
    const coverage = coverageMessage(condition, dataset(), data.metrics);
    status.textContent = `Ranked by ${metric.id}. ${coverage.text}`;
    status.classList.toggle("coverage-warning", coverage.warning);
  }
  datasetSelect.addEventListener("change", updatePatterns);
  patternSelect.addEventListener("change", updateAccelerations);
  accelerationSelect.addEventListener("change", updateTable);
  metricSelect.addEventListener("change", updateTable);
  updatePatterns();
}
