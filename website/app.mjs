import { el, fetchJSON } from "./dom.mjs";
import {
  experimentURL,
  parseRoute,
  validateRegistry,
  hasResults,
  resolveExperimentView,
} from "./registry.mjs";
import { validateRows } from "./stats.mjs";
import { mountQuantitative, QUANTITATIVE_MODULES } from "./experiments/quantitative.mjs";
import { mountMedia, validateMedia } from "./media.mjs";
import { renderProtocol } from "./protocol.mjs";

const main = document.getElementById("application");
const tooltip = document.getElementById("tooltip");
document.querySelector(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  main.focus();
  main.scrollIntoView();
});
const cache = new Map();
let registry;
let request = 0;

function pageHeading(title, description, dimension) {
  const header = el("header", null, { class: "page-heading" });
  if (dimension) header.append(el("p", dimension, { class: "eyebrow" }));
  header.append(el("h1", title, { tabindex: "-1" }));
  if (description) header.append(el("p", description, { class: "intro" }));
  return header;
}
function renderOverview() {
  const welcome = pageHeading("MRBench: Benchmarking Foundation Models for MRI Reconstruction");
  const message = el("p", null, { class: "intro welcome" });
  message.append(
    el("strong", "Welcome to MRBench!"),
    document.createTextNode(
      " Explore how foundation models for MRI reconstruction perform across different acquisition conditions, signal representations, and imaging domains. Select an experiment below to learn about its design and explore the available results.",
    ),
  );
  welcome.append(message);
  main.append(welcome);
  const section = el("section", null, { "aria-labelledby": "experiments-heading" });
  section.append(el("h2", "Explore the experiments", { id: "experiments-heading" }));
  const table = el("table", null, { class: "experiment-table" });
  table.append(
    el("caption", "Evaluation dimensions, experiments and available results", { class: "sr-only" }),
  );
  const head = el("thead");
  const header = el("tr");
  for (const name of ["Experiments", "Description", "Results"])
    header.append(el("th", name, { scope: "col" }));
  head.append(header);
  table.append(head);
  for (const dimension of registry.dimensions) {
    const body = el("tbody", null, { class: `dimension dimension-${dimension.tone}` });
    const groupRow = el("tr", null, { class: "dimension-heading" });
    groupRow.append(el("th", dimension.name, { colspan: "3", scope: "rowgroup" }));
    body.append(groupRow);
    for (const entry of registry.experiments.filter((e) => e.dimension === dimension.id)) {
      const row = el("tr", null, { "data-experiment": entry.id });
      const name = el("th", null, { scope: "row" });
      const link = el("a", null, { href: experimentURL(entry.id), class: "experiment-name" });
      link.append(
        el("span", null, { class: "experiment-bullet", "aria-hidden": "true" }),
        document.createTextNode(entry.name),
      );
      name.append(link);
      const description = el("td", null, { class: "experiment-description" });
      if (entry.description || entry.setup.length)
        description.append(
          el("a", "Details →", {
            href: experimentURL(entry.id),
            "aria-label": `${entry.name}: details`,
          }),
        );
      const results = el("td", null, { class: "experiment-results" });
      if (hasResults(entry))
        results.append(
          el("a", "View results →", {
            href: experimentURL(entry.id, true),
            class: "results-link",
            "aria-label": `${entry.name}: view results`,
          }),
        );
      else results.append(el("span", "Coming soon", { class: "coming-soon" }));
      row.append(name, description, results);
      body.append(row);
    }
    table.append(body);
  }
  section.append(table);
  main.append(section);
}
function renderSetup(root, experiment) {
  const setup = el("section", null, { class: "setup", "aria-labelledby": "setup-heading" });
  setup.append(el("h2", "Experiment setup", { id: "setup-heading" }));
  const descriptions = el("dl", null, { class: "setup-list" });
  for (const item of experiment.setup) {
    const group = el("div");
    group.append(el("dt", item.label), el("dd", item.text));
    descriptions.append(group);
  }
  setup.append(descriptions);
  root.append(setup);
}
async function metrics(sourceId) {
  if (!cache.has(sourceId)) {
    const source = registry.data_sources[sourceId];
    const pending = fetchJSON(source.metrics_path)
      .then(validateRows)
      .then((rows) => {
        if (
          source.metric_protocol !== "cmrbench_recon_metrics_v2_subject_p995" ||
          rows[0].metric_protocol !== source.metric_protocol
        )
          throw new Error(
            "Add protocol-specific labels before publishing a different metrics protocol",
          );
        return rows;
      })
      .catch((error) => {
        cache.delete(sourceId);
        throw error;
      });
    cache.set(sourceId, pending);
  }
  return cache.get(sourceId);
}
async function renderExperiment(route, token) {
  const experiment = registry.experiments.find((e) => e.id === route.id);
  if (!experiment) {
    renderNotFound();
    return;
  }
  const view = resolveExperimentView(experiment, route.view);
  if (view !== route.view) history.replaceState(null, "", experimentURL(experiment.id, view));
  main.append(el("a", "← All experiments", { href: "#/overview", class: "back-link" }));
  const dimension = registry.dimensions.find((d) => d.id === experiment.dimension);
  main.append(pageHeading(experiment.name, null, dimension.name));
  main.dataset.experimentView = view;
  const selector = el("nav", null, { class: "experiment-nav", "aria-label": "Experiment views" });
  for (const [key, label] of [
    ["details", "Experiment details"],
    ["results", "Results"],
  ]) {
    const item =
      key === "results" && !hasResults(experiment)
        ? el("span", label, {
            class: "disabled",
            "aria-disabled": "true",
            title: "Results coming soon",
          })
        : el("a", label, {
            href: experimentURL(experiment.id, key, route.params),
            "data-view": key,
          });
    if (view === key) {
      item.classList.add("active");
      item.setAttribute("aria-current", "page");
    }
    selector.append(item);
  }
  main.append(selector);
  if (view === "details") {
    const objective = el("section", null, {
      class: "objective",
      "aria-labelledby": "objective-heading",
    });
    objective.append(
      el("h2", "Experiment objective", { id: "objective-heading" }),
      el("p", experiment.objective || experiment.description),
    );
    main.append(objective);
    renderSetup(main, experiment);
    mountMedia(main, experiment.illustrations, "Experiment illustration", "illustrations");
    if (route.view === "results")
      main.append(el("p", "Results coming soon.", { class: "unavailable", role: "status" }));
    return;
  }
  for (const module of experiment.quantitative) {
    const loading = el("p", "Loading quantitative results…", { class: "loading", role: "status" });
    main.append(loading);
    try {
      const rows = await metrics(module.source);
      if (token !== request) return;
      loading.remove();
      mountQuantitative(main, {
        module,
        source: registry.data_sources[module.source],
        rows,
        params: route.params,
        onSelection(selection) {
          const params = new URLSearchParams(selection);
          history.replaceState(null, "", experimentURL(experiment.id, "results", params));
          for (const link of selector.querySelectorAll("a[data-view]"))
            link.href = experimentURL(experiment.id, link.dataset.view, params);
        },
      });
    } catch (error) {
      if (token !== request) return;
      loading.remove();
      report(error, main);
    }
  }
  if (token !== request) return;
  mountMedia(main, experiment.qualitative, "Qualitative results", "qualitative");
}
function renderNotFound() {
  main.append(
    pageHeading("Experiment not found", "This link does not identify a registered experiment."),
    el("a", "← All experiments", { href: "#/overview", class: "back-link" }),
  );
}
function report(error, root = main) {
  root.append(
    el(
      "p",
      `Unable to load this view: ${error.message}. Check the data files and serve the website over HTTP.`,
      { class: "error", role: "alert" },
    ),
  );
}
async function navigate({ focus = true } = {}) {
  const token = ++request;
  tooltip.hidden = true;
  main.replaceChildren();
  delete main.dataset.experimentView;
  const route = parseRoute(location.hash);
  for (const link of document.querySelectorAll(".masthead nav a")) {
    const active = link.dataset.page === route.page;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  window.scrollTo(0, 0);
  if (route.page === "overview") renderOverview();
  else if (route.page === "protocol") renderProtocol(main, pageHeading);
  else if (route.page === "experiment") await renderExperiment(route, token);
  else renderNotFound();
  if (token !== request) return;
  const h1 = main.querySelector("h1");
  document.title =
    route.page === "overview"
      ? "MRBench: Benchmarking Foundation Models for MRI Reconstruction"
      : `${h1?.textContent}${route.page === "experiment" ? ` — ${main.dataset.experimentView === "results" ? "Results" : "Experiment details"}` : ""} · MRBench`;
  if (focus) h1?.focus({ preventScroll: true });
}
try {
  registry = validateRegistry(await fetchJSON("data/experiments.json"));
  for (const experiment of registry.experiments) {
    for (const module of experiment.quantitative)
      if (!QUANTITATIVE_MODULES[module.type])
        throw new Error(`Unregistered quantitative module: ${module.type}`);
    validateMedia(experiment.illustrations);
    validateMedia(experiment.qualitative);
  }
  await navigate({ focus: false });
  window.addEventListener("hashchange", () => navigate().catch((error) => report(error)));
  window.addEventListener(
    "scroll",
    () => {
      tooltip.hidden = true;
    },
    { passive: true },
  );
} catch (error) {
  main.replaceChildren();
  report(error);
}
