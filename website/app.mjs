import { describe, mean, summarize, pairRows, validateRows } from "./stats.mjs";
const $ = (id) => document.getElementById(id);
const colors = { SDUM: "#446d9d", CardioMM: "#208977", RAM: "#c77853" };
const palette = ["#446d9d", "#208977", "#c77853", "#8267a3"];
const patternNames = {
  kt_uniform: "k-t uniform",
  kt_gaussian: "k-t Gaussian",
  kt_radial: "k-t pseudo-radial",
  uniform: "Uniform",
  vista: "VISTA",
  random: "Random",
  equispaced: "Equispaced",
  radial: "Pseudo-radial",
};
const state = {
  rows: [],
  experiments: [],
  cases: [],
  metric: "NRMSE",
  sort: "NRMSE",
  subject: null,
  request: 0,
};
const fmt = (x, m = "NRMSE") => Number(x).toFixed(m === "PSNR" ? 2 : m === "runtime_sec" ? 1 : 4);
function label(model) {
  const variant = state.rows.find((row) => row.model === model)?.variant_or_implementation;
  if (model === "SDUM" && variant) {
    return `SDUM ${variant[0].toUpperCase()}${variant.slice(1)}`;
  }
  if (model === "RAM" && variant) {
    return `RAM ${variant === "deepinv" ? "DeepInv" : variant}`;
  }
  return model;
}
const color = (m, i = 0) => colors[m] || palette[i % palette.length];
function el(tag, text, attrs = {}) {
  const node = document.createElement(tag);
  if (text !== null && text !== undefined) node.textContent = text;
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}
function option(select, value, text) {
  select.append(el("option", text, { value }));
}
function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
function add(svg, tag, attrs, text) {
  const node = svgEl(tag, attrs, text);
  svg.append(node);
  return node;
}
function svgChart(container, labelText) {
  container.replaceChildren();
  const svg = svgEl("svg", { viewBox: "0 0 540 310", role: "img", "aria-label": labelText });
  container.append(svg);
  return svg;
}
function range(values, pad = 0.1) {
  let low = Math.min(...values),
    high = Math.max(...values);
  const width = high - low || Math.max(Math.abs(low) * 0.1, 0.01);
  return [low - width * pad, high + width * pad];
}
function scale(domain, start, end) {
  return (x) => start + ((x - domain[0]) / (domain[1] - domain[0])) * (end - start);
}
function axis(svg, domain, metric, { xAxis = false, lo = 46, hi = 500 } = {}) {
  const y = scale(domain, 255, 25);
  for (let i = 0; i <= 4; i++) {
    const value = domain[0] + ((domain[1] - domain[0]) * i) / 4;
    if (!xAxis) {
      add(svg, "line", { x1: lo, x2: hi, y1: y(value), y2: y(value), stroke: "#e8edf1" });
      add(
        svg,
        "text",
        { x: lo - 10, y: y(value) + 4, "text-anchor": "end", fill: "#798698", "font-size": 10 },
        fmt(value, metric),
      );
    }
  }
  add(svg, "line", { x1: lo, x2: hi, y1: 255, y2: 255, stroke: "#ccd6df" });
  return y;
}
function showTip(node, text) {
  node.setAttribute("aria-label", text);
  node.addEventListener("pointerenter", (event) => {
    const tip = $("tooltip");
    tip.textContent = text;
    tip.hidden = false;
    positionTip(event.clientX, event.clientY);
  });
  node.addEventListener("pointermove", (event) => positionTip(event.clientX, event.clientY));
  node.addEventListener("pointerleave", () => {
    $("tooltip").hidden = true;
  });
  node.addEventListener("focus", () => {
    const rect = node.getBoundingClientRect();
    $("tooltip").textContent = text;
    $("tooltip").hidden = false;
    positionTip(rect.x, rect.y);
  });
  node.addEventListener("blur", () => {
    $("tooltip").hidden = true;
  });
  node.addEventListener("click", () => {
    state.subject = node.dataset.subject;
    renderResults();
  });
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      state.subject = node.dataset.subject;
      renderResults();
    }
  });
}
function positionTip(x, y) {
  const tip = $("tooltip");
  tip.style.left = `${Math.min(x + 12, window.innerWidth - tip.offsetWidth - 12)}px`;
  tip.style.top = `${Math.max(8, Math.min(y + 12, window.innerHeight - tip.offsetHeight - 10))}px`;
}
function jitter(id) {
  let hash = 2166136261;
  for (const ch of id) hash = Math.imul(hash ^ ch.charCodeAt(0), 16777619);
  return (((hash >>> 0) % 1000) / 999 - 0.5) * 72;
}
function samplingRows() {
  const [sampling, implementation] = $("sampling").value.split("|");
  return state.rows.filter(
    (r) => r.sampling === sampling && r.sampling_implementation === implementation,
  );
}
function currentRows() {
  return samplingRows().filter((r) => r.requested_R === Number($("acceleration").value));
}
function models() {
  return [...new Set(state.rows.map((r) => r.model))].sort(
    (a, b) => ["SDUM", "CardioMM", "RAM"].indexOf(a) - ["SDUM", "CardioMM", "RAM"].indexOf(b),
  );
}

function distribution(rows) {
  const metric = state.metric;
  const names = models();
  const svg = svgChart(
    $("distribution"),
    `${metric} subject distributions for selected acquisition`,
  );
  const domain = range(rows.map((r) => r[metric]));
  const y = axis(svg, domain, metric);
  names.forEach((model, index) => {
    const group = rows.filter((r) => r.model === model);
    if (!group.length) return;
    const center = 108 + index * (370 / Math.max(1, names.length - 1));
    const summary = describe(group.map((r) => r[metric]));
    const shade = color(model, index);
    add(svg, "line", {
      x1: center,
      x2: center,
      y1: y(summary.min),
      y2: y(summary.max),
      stroke: shade,
      "stroke-width": 1.3,
    });
    for (const value of [summary.min, summary.max])
      add(svg, "line", {
        x1: center - 9,
        x2: center + 9,
        y1: y(value),
        y2: y(value),
        stroke: shade,
      });
    add(svg, "rect", {
      x: center - 21,
      y: y(summary.q3),
      width: 42,
      height: Math.max(2, y(summary.q1) - y(summary.q3)),
      fill: shade,
      "fill-opacity": 0.13,
      stroke: shade,
      "stroke-width": 1.2,
    });
    add(svg, "line", {
      x1: center - 21,
      x2: center + 21,
      y1: y(summary.median),
      y2: y(summary.median),
      stroke: shade,
      "stroke-width": 2,
    });
    for (const row of group) {
      const selected = row.subject === state.subject;
      const dot = add(svg, "circle", {
        cx: center + jitter(row.subject),
        cy: y(row[metric]),
        r: selected ? 5 : 2.9,
        fill: shade,
        "fill-opacity": selected ? 1 : 0.58,
        stroke: selected ? "#203149" : "none",
        "stroke-width": 1.5,
        tabindex: 0,
      });
      dot.dataset.subject = row.subject;
      showTip(dot, `${row.subject} · ${label(model)} · ${metric} ${fmt(row[metric], metric)}`);
    }
    add(
      svg,
      "text",
      { x: center, y: 280, "text-anchor": "middle", fill: "#203149", "font-size": 11 },
      label(model),
    );
    add(
      svg,
      "text",
      { x: center, y: 298, "text-anchor": "middle", fill: "#798698", "font-size": 10 },
      `n = ${group.length}`,
    );
  });
}

function trend() {
  const rows = samplingRows(),
    metric = state.metric;
  const rs = [...new Set(rows.map((r) => r.requested_R))].sort((a, b) => a - b);
  const groupSummaries = [];
  models().forEach((model, i) => {
    const points = rs
      .map((r) => ({ r, rows: rows.filter((row) => row.model === model && row.requested_R === r) }))
      .filter((p) => p.rows.length)
      .map((p) => ({ ...p, summary: describe(p.rows.map((row) => row[metric])) }));
    groupSummaries.push({ model, i, points });
  });
  const domain = range(
    groupSummaries.flatMap((g) =>
      g.points.flatMap((p) => [p.summary.mean - p.summary.sd, p.summary.mean + p.summary.sd]),
    ),
  );
  const svg = svgChart($("trend"), `${metric} mean and subject SD across nominal accelerations`),
    y = axis(svg, domain, metric);
  const x = scale(range(rs, 0.1), 66, 495);
  const selectedR = Number($("acceleration").value);
  add(svg, "line", {
    x1: x(selectedR),
    x2: x(selectedR),
    y1: 20,
    y2: 255,
    stroke: "#bdcad5",
    "stroke-dasharray": "3 5",
  });
  for (const group of groupSummaries) {
    const shade = color(group.model, group.i);
    add(svg, "polyline", {
      points: group.points.map((p) => `${x(p.r)} ${y(p.summary.mean)}`).join(" "),
      stroke: shade,
      "stroke-width": 2,
      fill: "none",
    });
    for (const p of group.points) {
      const px = x(p.r) + (group.i - 1) * 4;
      add(svg, "line", {
        x1: px,
        x2: px,
        y1: y(p.summary.mean - p.summary.sd),
        y2: y(p.summary.mean + p.summary.sd),
        stroke: shade,
        "stroke-opacity": 0.5,
      });
      for (const value of [p.summary.mean - p.summary.sd, p.summary.mean + p.summary.sd])
        add(svg, "line", {
          x1: px - 4,
          x2: px + 4,
          y1: y(value),
          y2: y(value),
          stroke: shade,
          "stroke-opacity": 0.5,
        });
      const dot = add(svg, "circle", { cx: x(p.r), cy: y(p.summary.mean), r: 4, fill: shade });
      add(
        dot,
        "title",
        {},
        `${label(group.model)} · nominal R${p.r} · mean ${fmt(p.summary.mean, metric)} · SD ${fmt(p.summary.sd, metric)}`,
      );
    }
  }
  for (const r of rs)
    add(
      svg,
      "text",
      { x: x(r), y: 279, "text-anchor": "middle", fill: "#65758a", "font-size": 11 },
      `R = ${r}`,
    );
  add(
    svg,
    "text",
    { x: 280, y: 303, "text-anchor": "middle", fill: "#798698", "font-size": 10 },
    "Nominal acceleration (actual R is reported separately)",
  );
  $("legend").replaceChildren(
    ...groupSummaries.map((g) => {
      const span = el("span");
      span.append(
        el("i", null, { class: "swatch", style: `background:${color(g.model, g.i)}` }),
        document.createTextNode(label(g.model)),
      );
      return span;
    }),
  );
}

function samplingComparison() {
  const metric = state.metric,
    r = Number($("acceleration").value);
  const rows = state.rows.filter((row) => row.requested_R === r);
  const choices = [...new Set(rows.map((row) => `${row.sampling}|${row.sampling_implementation}`))];
  const series = models().map((model, i) => ({
    model,
    i,
    points: choices
      .map((choice) => ({
        choice,
        group: rows.filter(
          (row) =>
            row.model === model && `${row.sampling}|${row.sampling_implementation}` === choice,
        ),
      }))
      .filter((point) => point.group.length)
      .map((point) => ({ ...point, summary: describe(point.group.map((row) => row[metric])) })),
  }));
  const domain = range(
    series.flatMap((s) =>
      s.points.flatMap((p) => [p.summary.mean - p.summary.sd, p.summary.mean + p.summary.sd]),
    ),
  );
  const svg = svgChart(
    $("sampling-comparison"),
    `${metric} across physical sampling families at nominal R${r}`,
  );
  const y = axis(svg, domain, metric);
  const xs = choices.map((_, i) => 106 + i * (370 / Math.max(1, choices.length - 1)));
  for (const s of series) {
    const shade = color(s.model, s.i);
    add(svg, "polyline", {
      points: s.points
        .map((p) => `${xs[choices.indexOf(p.choice)]} ${y(p.summary.mean)}`)
        .join(" "),
      fill: "none",
      stroke: shade,
      "stroke-width": 2,
    });
    for (const p of s.points) {
      const px = xs[choices.indexOf(p.choice)] + (s.i - 1) * 5;
      add(svg, "line", {
        x1: px,
        x2: px,
        y1: y(p.summary.mean - p.summary.sd),
        y2: y(p.summary.mean + p.summary.sd),
        stroke: shade,
        "stroke-opacity": 0.6,
      });
      for (const value of [p.summary.mean - p.summary.sd, p.summary.mean + p.summary.sd])
        add(svg, "line", { x1: px - 4, x2: px + 4, y1: y(value), y2: y(value), stroke: shade });
      const dot = add(svg, "circle", {
        cx: xs[choices.indexOf(p.choice)],
        cy: y(p.summary.mean),
        r: 4,
        fill: shade,
      });
      add(
        dot,
        "title",
        {},
        `${label(s.model)} · ${p.choice.split("|")[0]} · mean ${fmt(p.summary.mean, metric)} · SD ${fmt(p.summary.sd, metric)}`,
      );
    }
  }
  choices.forEach((choice, i) => {
    const group = rows.filter((row) => `${row.sampling}|${row.sampling_implementation}` === choice);
    const actual = describe([...new Map(group.map((row) => [row.subject, row.actual_R])).values()]);
    add(
      svg,
      "text",
      { x: xs[i], y: 280, "text-anchor": "middle", fill: "#203149", "font-size": 11 },
      patternNames[choice.split("|")[0]] || choice.split("|")[0],
    );
    add(
      svg,
      "text",
      { x: xs[i], y: 298, "text-anchor": "middle", fill: "#798698", "font-size": 10 },
      `actual R median ${actual.median.toFixed(2)}`,
    );
  });
}

function summaryTable(rows) {
  const summaries = summarize(rows);
  const sort = state.sort;
  const ordered = [...summaries].sort(
    (a, b) => (sort === "NRMSE" ? 1 : -1) * (a[sort].mean - b[sort].mean),
  );
  const best = {};
  for (const metric of ["NRMSE", "PSNR", "SSIM"])
    best[metric] = (metric === "NRMSE" ? Math.min : Math.max)(
      ...summaries.map((s) => s[metric].mean),
    );
  $("summary-body").replaceChildren();
  for (const item of ordered) {
    const tr = el("tr");
    const td = el("td");
    td.append(
      el("i", null, { class: "model-dot", style: `background:${color(item.model)}` }),
      document.createTextNode(label(item.model)),
    );
    tr.append(td, el("td", String(item.n)));
    for (const metric of ["NRMSE", "PSNR", "SSIM"])
      tr.append(
        el("td", `${fmt(item[metric].mean, metric)} ± ${fmt(item[metric].sd, metric)}`, {
          class: item[metric].mean === best[metric] ? "best" : "",
        }),
      );
    tr.append(el("td", fmt(item.runtime_sec.median, "runtime_sec")));
    $("summary-body").append(tr);
  }
  return summaries;
}

function paired(rows) {
  const a = $("model-a").value,
    b = $("model-b").value,
    metric = state.metric;
  const pairs = pairRows(rows, a, b);
  const svg = svgChart($("paired"), `Paired ${metric}: ${label(a)} versus ${label(b)}`);
  if (!pairs.length) {
    add(
      svg,
      "text",
      { x: 270, y: 145, "text-anchor": "middle", fill: "#65758a" },
      "No matched subjects available",
    );
    $("pair-stats").textContent = "No matched subjects.";
    return;
  }
  const domain = range(pairs.flatMap((p) => [p.a[metric], p.b[metric]]));
  const x = scale(domain, 70, 500),
    y = scale(domain, 255, 25);
  for (let i = 0; i <= 4; i++) {
    const value = domain[0] + ((domain[1] - domain[0]) * i) / 4;
    add(svg, "line", { x1: 70, x2: 500, y1: y(value), y2: y(value), stroke: "#e8edf1" });
    add(svg, "line", { x1: x(value), x2: x(value), y1: 25, y2: 255, stroke: "#e8edf1" });
    add(
      svg,
      "text",
      { x: 60, y: y(value) + 4, "text-anchor": "end", fill: "#798698", "font-size": 10 },
      fmt(value, metric),
    );
    add(
      svg,
      "text",
      { x: x(value), y: 273, "text-anchor": "middle", fill: "#798698", "font-size": 10 },
      fmt(value, metric),
    );
  }
  add(svg, "line", {
    x1: x(domain[0]),
    x2: x(domain[1]),
    y1: y(domain[0]),
    y2: y(domain[1]),
    stroke: "#94a3b4",
    "stroke-dasharray": "5 5",
  });
  for (const p of pairs) {
    const selected = p.subject === state.subject;
    const dot = add(svg, "circle", {
      cx: x(p.a[metric]),
      cy: y(p.b[metric]),
      r: selected ? 6 : 3.8,
      fill: selected ? "#203149" : color(b),
      "fill-opacity": selected ? 1 : 0.65,
      tabindex: 0,
    });
    dot.dataset.subject = p.subject;
    showTip(
      dot,
      `${p.subject} · ${a}: ${fmt(p.a[metric], metric)} · ${b}: ${fmt(p.b[metric], metric)}`,
    );
  }
  add(
    svg,
    "text",
    { x: 287, y: 302, "text-anchor": "middle", fill: "#65758a", "font-size": 11 },
    `Model A · ${label(a)} · ${metric}`,
  );
  add(
    svg,
    "text",
    {
      x: 16,
      y: 145,
      transform: "rotate(-90 16 145)",
      "text-anchor": "middle",
      fill: "#65758a",
      "font-size": 11,
    },
    `Model B · ${label(b)} · ${metric}`,
  );
  const diffs = pairs.map((p) =>
    metric === "NRMSE" ? p.a[metric] - p.b[metric] : p.b[metric] - p.a[metric],
  );
  const wins = diffs.filter((x) => x > 0).length;
  const avg = mean(diffs);
  $("pair-stats").replaceChildren(
    el("strong", `${avg >= 0 ? "+" : ""}${fmt(avg, metric)}${metric === "PSNR" ? " dB" : ""}`),
    el(
      "p",
      `Mean paired improvement of B over A. Positive favors B. B is better in ${wins}/${pairs.length} paired subjects. This is a descriptive comparison.`,
    ),
  );
  $("selected-subject").textContent = state.subject
    ? `Highlighted: ${state.subject}`
    : "No subject highlighted.";
}

function renderResults() {
  if (!state.rows.length) return;
  $("tooltip").hidden = true;
  const rows = currentRows();
  if (!rows.length) throw new Error("No results for selected condition");
  const actual = describe([...new Map(rows.map((r) => [r.subject, r.actual_R])).values()]);
  const first = rows[0];
  $("condition-detail").textContent =
    `Actual R: median ${actual.median.toFixed(2)} (range ${actual.min.toFixed(2)}–${actual.max.toFixed(2)}) · ACS ${first.ncalib ?? "not recorded"} · seed ${first.seed ?? "not recorded"} · ${first.sampling_implementation} implementation`;
  $("audit-state").textContent = "Matching physical-mask provenance";
  $("direction").textContent = state.metric === "NRMSE" ? "Lower is better" : "Higher is better";
  distribution(rows);
  trend();
  samplingComparison();
  summaryTable(rows);
  paired(rows);
}

async function loadMetrics(experiment) {
  const token = ++state.request;
  $("error").hidden = true;
  const entry = state.experiments.find((e) => e.id === experiment);
  if (!entry?.metrics_path) throw new Error("Experiment has no quantitative table");
  const rows = validateRows(await fetchJSON(entry.metrics_path));
  if (token !== state.request) return;
  if (rows[0].metric_protocol !== "cmrbench_recon_metrics_v2_subject_p995")
    throw new Error(
      "This viewer currently requires Metrics v2; add protocol-specific labels before publishing another protocol",
    );
  state.rows = rows;
  state.subject = null;
  const choices = [...new Set(rows.map((r) => `${r.sampling}|${r.sampling_implementation}`))];
  $("sampling").replaceChildren();
  for (const choice of choices) {
    const [method, impl] = choice.split("|");
    option($("sampling"), choice, `${patternNames[method] || method} · ${impl}`);
  }
  if (choices.includes("kt_uniform|cardiomm")) $("sampling").value = "kt_uniform|cardiomm";
  refreshAccelerations();
  for (const id of ["model-a", "model-b"]) {
    $(id).replaceChildren();
    for (const name of models()) option($(id), name, label(name));
  }
  $("model-a").value = models()[0];
  $("model-b").value = models()[1] || models()[0];
  const subjects = new Set(rows.map((r) => r.subject));
  const groups = new Set(
    rows.map((r) => `${r.model}|${r.sampling}|${r.sampling_implementation}|${r.requested_R}`),
  );
  $("n-subjects").textContent = subjects.size;
  $("n-models").textContent = models().length;
  $("n-conditions").textContent = groups.size;
  $("n-records").textContent = rows.length.toLocaleString("en-US");
  const download = document.querySelector("#results .section-heading a");
  download.href = entry.csv_path || entry.metrics_path;
  renderResults();
}
function refreshAccelerations() {
  const old = $("acceleration").value;
  $("acceleration").replaceChildren();
  const choices = [...new Set(samplingRows().map((r) => r.requested_R))].sort((a, b) => a - b);
  for (const r of choices) option($("acceleration"), r, `R = ${r}`);
  if (choices.includes(Number(old))) $("acceleration").value = old;
}

function renderExperiments() {
  const categories = [...new Set(state.experiments.map((e) => e.category))];
  const root = $("experiment-list");
  const statuses = {
    quantitative: "Cohort metrics",
    qualitative: "Qualitative example",
    implemented: "Code available",
    planned: "Planned",
    pending: "Results pending",
  };
  for (const category of categories) {
    root.append(el("h3", category, { class: "experiment-category" }));
    const group = el("div");
    for (const entry of state.experiments.filter((e) => e.category === category)) {
      const article = el("article", null, { class: "experiment-row" });
      const content = el("div");
      content.append(el("h3", entry.label), el("p", entry.description));
      const details = el("details");
      details.append(
        el("summary", "Settings and current evidence"),
        el("p", entry.settings),
        el("p", entry.limitations),
      );
      content.append(details);
      if (entry.metrics_path) {
        const button = el("button", "Explore metrics →", { class: "text-button" });
        button.addEventListener("click", async () => {
          try {
            $("experiment").value = entry.id;
            await loadMetrics(entry.id);
            activate("results");
          } catch (err) {
            report(err);
          }
        });
        content.append(button);
      } else if (entry.case_family) {
        const button = el("button", "Open archived cine →", { class: "text-button" });
        button.addEventListener("click", () => {
          $("case-sampling").value = entry.case_family;
          renderCases();
          activate("cases");
        });
        content.append(button);
      }
      article.append(
        content,
        el("span", statuses[entry.status], { class: `evidence ${entry.status}` }),
      );
      group.append(article);
    }
    root.append(group);
  }
}
function renderCases() {
  const filtered = state.cases.filter(
    (c) =>
      ($("case-sampling").value === "all" || c.sampling === $("case-sampling").value) &&
      ($("case-model").value === "all" || c.model === $("case-model").value),
  );
  $("case-count").textContent =
    `${filtered.length} archived animation${filtered.length === 1 ? "" : "s"} · candidate only · physical pairing not verified`;
  $("case-grid").replaceChildren();
  for (const item of filtered) {
    const card = el("article", null, { class: "case-card" });
    const screen = el("div", null, { class: "cine-screen" });
    const link = el("a", null, { href: item.asset, target: "_blank", rel: "noopener" });
    link.append(
      el("img", null, {
        src: item.asset,
        alt: `${item.case}, ${label(item.model)}, ${item.sampling}, nominal R${item.nominal_R}; candidate cine`,
        loading: "lazy",
      }),
    );
    screen.append(link);
    const info = el("div", null, { class: "case-info" });
    info.append(
      el("h3", label(item.model)),
      el("p", `${patternNames[item.sampling] || item.sampling} · nominal R${item.nominal_R}`),
      el(
        "p",
        `Source: ${item.implementation || "not recorded"} · configured ACS: ${item.ncalib ?? "not recorded"}`,
        { class: "asset-info" },
      ),
      el("p", `${item.case} · ${item.frames} encoded GIF frames · actual R not linked`, {
        class: "asset-info",
      }),
    );
    const details = el("details", null, { class: "asset-info" });
    details.append(
      el("summary", "Display and provenance"),
      el("p", item.display),
      el("p", item.condition_provenance),
    );
    info.append(details);
    card.append(screen, info);
    $("case-grid").append(card);
  }
}
function activate(tab) {
  const allowed = ["results", "experiments", "cases", "protocol"];
  if (!allowed.includes(tab)) tab = "results";
  for (const item of allowed) $(item).hidden = item !== tab;
  document.querySelectorAll(".masthead [data-tab]").forEach((a) => {
    a.classList.toggle("active", a.dataset.tab === tab);
    if (a.dataset.tab === tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  if (location.hash !== `#${tab}`) history.replaceState(null, "", `#${tab}`);
}
function report(error) {
  $("error").textContent =
    `Result loading failed: ${error.message}. Serve this directory with a local HTTP server; direct file URLs cannot load the tables.`;
  $("error").hidden = false;
  $("loading").hidden = true;
}
async function fetchJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}
function downloadSummary() {
  const rows = summarize(currentRows());
  const condition = currentRows()[0];
  const header = [
    "model",
    "subjects",
    "sampling",
    "sampling_implementation",
    "nominal_R",
    "metric_protocol",
    "NRMSE_mean",
    "NRMSE_SD",
    "PSNR_mean",
    "PSNR_SD",
    "SSIM_mean",
    "SSIM_SD",
    "runtime_median_seconds",
  ];
  const values = rows.map((r) => [
    r.model,
    r.n,
    condition.sampling,
    condition.sampling_implementation,
    condition.requested_R,
    condition.metric_protocol,
    r.NRMSE.mean,
    r.NRMSE.sd,
    r.PSNR.mean,
    r.PSNR.sd,
    r.SSIM.mean,
    r.SSIM.sd,
    r.runtime_sec.median,
  ]);
  const csv = [header, ...values]
    .map((row) => row.map((v) => `"${String(v).replaceAll('"', '""')}"`).join(","))
    .join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const a = el("a", null, {
    href: url,
    download: `${condition.experiment_id}_${condition.sampling}_R${condition.requested_R}_summary.csv`,
  });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function init() {
  try {
    const [experiments, cases] = await Promise.all([
      fetchJSON("data/experiments.json"),
      fetchJSON("data/cases.json"),
    ]);
    state.experiments = experiments;
    state.cases = cases;
    for (const e of experiments.filter((e) => e.metrics_path))
      option($("experiment"), e.id, e.label);
    for (const method of [...new Set(cases.map((c) => c.sampling))])
      option($("case-sampling"), method, patternNames[method] || method);
    for (const model of [...new Set(cases.map((c) => c.model))])
      option($("case-model"), model, label(model));
    await loadMetrics($("experiment").value);
    renderExperiments();
    renderCases();
    $("loading").hidden = true;
    $("application").hidden = false;
    activate(location.hash.slice(1));
    $("experiment").addEventListener("change", () =>
      loadMetrics($("experiment").value).catch(report),
    );
    $("sampling").addEventListener("change", () => {
      state.subject = null;
      refreshAccelerations();
      renderResults();
    });
    $("acceleration").addEventListener("change", () => {
      state.subject = null;
      renderResults();
    });
    $("metric").addEventListener("change", () => {
      state.metric = $("metric").value;
      state.sort = state.metric;
      renderResults();
    });
    for (const id of ["model-a", "model-b"])
      $(id).addEventListener("change", () => paired(currentRows()));
    for (const id of ["case-sampling", "case-model"]) $(id).addEventListener("change", renderCases);
    document.querySelectorAll("[data-sort]").forEach((button) =>
      button.addEventListener("click", () => {
        state.sort = button.dataset.sort;
        summaryTable(currentRows());
      }),
    );
    document.querySelectorAll("[data-tab]").forEach((a) =>
      a.addEventListener("click", (event) => {
        event.preventDefault();
        activate(a.dataset.tab);
      }),
    );
    window.addEventListener("hashchange", () => activate(location.hash.slice(1)));
    $("clear-subject").addEventListener("click", () => {
      state.subject = null;
      renderResults();
    });
    $("download-summary").addEventListener("click", downloadSummary);
  } catch (error) {
    report(error);
  }
}
await init();
