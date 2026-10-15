import { el } from "../dom.mjs";
import { describe } from "../stats.mjs";
import { modelColor, modelLabel, modelNames } from "../models.mjs";
import { createCharts } from "../charts.mjs";
import { acceleration } from "./acceleration.mjs";
import { sampling } from "./sampling.mjs";
import {
  METRICS,
  patterns,
  patternLabel,
  accelerations,
  selectCondition,
  initialSelection,
} from "./conditions.mjs";

// Each entry owns its comparison module; future experiments can register other renderers.
export const QUANTITATIVE_MODULES = Object.freeze({ acceleration, sampling });

function filter(id, label, choices, selected) {
  const wrapper = el("label", label, { for: id });
  const select = el("select", null, { id });
  for (const [value, text] of choices) select.append(el("option", text, { value }));
  select.value = String(selected);
  wrapper.append(select);
  return wrapper;
}
function chartCard(title, container, caption) {
  const card = el("article", null, { class: "chart-card" });
  card.append(
    el("h3", title),
    el("div", null, { id: container, class: "chart" }),
    el("p", caption, { class: "chart-caption" }),
  );
  return card;
}

export function mountQuantitative(root, { module, source, rows, params, onSelection }) {
  const definition = QUANTITATIVE_MODULES[module.type];
  if (!definition) throw new Error(`Unsupported quantitative module: ${module.type}`);
  const selection = initialSelection(rows, params);
  const state = { rows, metric: selection.metric, subject: null };
  const section = el("section", null, {
    id: "quantitative",
    "aria-labelledby": "quantitative-title",
  });
  const heading = el("div", null, { class: "section-heading" });
  heading.append(el("h2", "Quantitative results", { id: "quantitative-title" }));
  const downloads = el("div", null, { class: "data-links" });
  downloads.append(
    el("a", "Download data (CSV)", { href: source.csv_path, download: "" }),
    el("a", "Audit (JSON)", { href: source.audit_path, download: "" }),
  );
  heading.append(downloads);
  const cohort = el(
    "p",
    `${source.cohort} · ${new Set(rows.map((r) => r.subject)).size} matched subjects per condition · ${modelNames(
      rows,
    )
      .map((m) => modelLabel(m, rows))
      .join(", ")}.`,
    { class: "cohort-note" },
  );
  const controls = el("div", null, { class: "controls" });
  const samplingFilter = filter(
    "sampling",
    "Sampling pattern",
    patterns(rows).map((p) => [p, patternLabel(p)]),
    selection.sampling,
  );
  const rFilter = filter(
    "acceleration",
    "Nominal acceleration",
    accelerations(rows, selection.sampling).map((r) => [r, `R = ${r}`]),
    selection.acceleration,
  );
  const metricFilter = filter(
    "metric",
    "Quality metric",
    METRICS.map((m) => [m, m === "PSNR" ? "PSNR (dB)" : m]),
    selection.metric,
  );
  controls.append(
    ...(module.type === "sampling"
      ? [rFilter, samplingFilter, metricFilter]
      : [samplingFilter, rFilter, metricFilter]),
  );
  const context = el("div", null, { class: "condition-context" });
  const actual = el("p", null, { id: "condition-detail" });
  const direction = el("span", null, { id: "direction" });
  context.append(actual, direction);
  const legend = el("div", null, { class: "legend", "aria-label": "Models" });
  for (const model of modelNames(rows)) {
    const entry = el("span", null, { "data-model": model });
    entry.append(
      el("i", null, { class: "swatch", style: `background:${modelColor(model)}` }),
      document.createTextNode(modelLabel(model, rows)),
    );
    legend.append(entry);
  }
  const chartsRoot = el("div", null, { class: "charts" });
  chartsRoot.append(
    chartCard(
      "Subject distribution",
      "distribution",
      "Dots: subjects. Box: median and interquartile range; whiskers: observed range. Select a dot to highlight that subject across models.",
    ),
    chartCard(definition.title, definition.container, definition.caption),
  );
  const subjectNote = el("div", null, { class: "subject-note" });
  const selected = el("span", "No subject selected.");
  const clear = el("button", "Clear selection", { type: "button", class: "text-button" });
  clear.hidden = true;
  subjectNote.append(selected, clear);
  const mobileNote = el("p", "Swipe horizontally within a plot to inspect it.", {
    class: "mobile-note",
  });
  const provenance = el("details", null, { class: "provenance" });
  provenance.append(
    el("summary", "Comparison and mask provenance"),
    el("p", source.context),
    el(
      "p",
      "Actual R includes acquired ACS. Subjects are matched across models and mask hashes are audited. Error bars show between-subject sample SD, not a confidence interval.",
    ),
  );
  section.append(
    heading,
    cohort,
    controls,
    context,
    legend,
    chartsRoot,
    mobileNote,
    subjectNote,
    provenance,
  );
  root.append(section);
  const samplingSelect = samplingFilter.querySelector("select");
  const rSelect = rFilter.querySelector("select");
  const metricSelect = metricFilter.querySelector("select");
  const charts = createCharts(state, render);

  function render() {
    const current = selectCondition(rows, samplingSelect.value, rSelect.value);
    if (!current.length) throw new Error("No results for the selected condition");
    const actualStats = describe([
      ...new Map(current.map((r) => [r.subject, r.actual_R])).values(),
    ]);
    const first = current[0];
    actual.textContent = `Selected condition · actual R: median ${actualStats.median.toFixed(2)} (range ${actualStats.min.toFixed(2)}–${actualStats.max.toFixed(2)}) · ACS ${first.ncalib} · seed ${first.seed} · ${first.sampling_implementation} implementation`;
    direction.textContent = `${state.metric}${state.metric === "PSNR" ? " (dB)" : ""} · ${state.metric === "NRMSE" ? "Lower" : "Higher"} is better`;
    charts.distribution(current);
    definition.draw(charts);
    selected.textContent = state.subject
      ? `Highlighted subject: ${state.subject}`
      : "No subject selected.";
    clear.hidden = !state.subject;
    onSelection({ sampling: samplingSelect.value, R: rSelect.value, metric: state.metric });
  }
  samplingSelect.addEventListener("change", () => {
    const oldR = Number(rSelect.value);
    const rs = accelerations(rows, samplingSelect.value);
    rSelect.replaceChildren(...rs.map((r) => el("option", `R = ${r}`, { value: r })));
    if (rs.includes(oldR)) rSelect.value = String(oldR);
    state.subject = null;
    render();
  });
  rSelect.addEventListener("change", () => {
    state.subject = null;
    render();
  });
  metricSelect.addEventListener("change", () => {
    state.metric = metricSelect.value;
    render();
  });
  clear.addEventListener("click", () => {
    state.subject = null;
    render();
  });
  render();
}
