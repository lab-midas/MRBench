import { describe } from "./stats.mjs";
import { modelColor, modelLabel, modelNames } from "./models.mjs";
import { patternNames, patternLabel, patterns } from "./experiments/conditions.mjs";
export function createCharts(state, renderResults) {
  const $ = (id) => document.getElementById(id);
  const fmt = (x, m = "NRMSE") => Number(x).toFixed(m === "PSNR" ? 2 : 4);
  const color = modelColor;
  const label = (model) => modelLabel(model, state.rows);
  const models = () => modelNames(state.rows);
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
    const tip = $("tooltip");
    const open = (x, y) => {
      tip.textContent = text;
      tip.hidden = false;
      tip.style.left = Math.max(8, Math.min(x + 12, innerWidth - tip.offsetWidth - 12)) + "px";
      tip.style.top = Math.max(8, Math.min(y + 12, innerHeight - tip.offsetHeight - 10)) + "px";
    };
    node.addEventListener("pointerenter", (e) => open(e.clientX, e.clientY));
    node.addEventListener("pointermove", (e) => open(e.clientX, e.clientY));
    node.addEventListener("pointerleave", () => {
      tip.hidden = true;
    });
    node.addEventListener("focus", () => {
      const r = node.getBoundingClientRect();
      open(r.x, r.y);
    });
    node.addEventListener("blur", () => {
      tip.hidden = true;
    });
    const select = () => {
      if (!node.dataset.subject) return;
      state.subject = node.dataset.subject;
      tip.hidden = true;
      renderResults();
      // Restore focus to the corresponding rebuilt point for keyboard users.
      document
        .querySelector('#distribution [data-subject="' + state.subject + '"]')
        ?.focus({ preventScroll: true });
    };
    node.addEventListener("click", select);
    node.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && node.dataset.subject) {
        e.preventDefault();
        select();
      }
    });
  }
  function summaryTip(model, rows, metric, condition) {
    const stats = describe(rows.map((r) => r[metric]));
    const actual = describe(rows.map((r) => r.actual_R));
    return (
      label(model) +
      " · " +
      condition +
      " · " +
      metric +
      " mean " +
      fmt(stats.mean, metric) +
      " · sample SD " +
      fmt(stats.sd, metric) +
      " · n = " +
      stats.n +
      " · actual R median " +
      actual.median.toFixed(2) +
      " (range " +
      actual.min.toFixed(2) +
      "–" +
      actual.max.toFixed(2) +
      ")"
    );
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
          "data-model": model,
          "fill-opacity": selected ? 1 : 0.58,
          stroke: selected ? "#203149" : "none",
          "stroke-width": 1.5,
          tabindex: 0,
        });
        dot.dataset.subject = row.subject;
        showTip(
          dot,
          `${row.subject} · ${label(model)} · ${metric} ${fmt(row[metric], metric)} · nominal R${row.requested_R}, actual R ${row.actual_R.toFixed(2)} · ${row.sampling_implementation} · mask ${row.binary_mask_sha256.slice(0, 12)}…`,
        );
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
        .map((r) => ({
          r,
          rows: rows.filter((row) => row.model === model && row.requested_R === r),
        }))
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
        points: group.points
          .map((p) => `${x(p.r) + (group.i - 1) * 4} ${y(p.summary.mean)}`)
          .join(" "),
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
        const dot = add(svg, "circle", {
          cx: px,
          cy: y(p.summary.mean),
          r: 4,
          fill: shade,
          "data-model": group.model,
          tabindex: 0,
        });
        showTip(dot, summaryTip(group.model, p.rows, metric, `nominal R${p.r}`));
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
  }

  function samplingComparison() {
    const metric = state.metric,
      r = Number($("acceleration").value);
    const rows = state.rows.filter((row) => row.requested_R === r);
    const choices = patterns(rows);
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
      for (const p of s.points) {
        const px = xs[choices.indexOf(p.choice)] + (s.i - (series.length - 1) / 2) * 16;
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
          cx: px,
          cy: y(p.summary.mean),
          r: 4,
          fill: shade,
          "data-model": s.model,
          tabindex: 0,
        });
        showTip(
          dot,
          summaryTip(s.model, p.group, metric, `${patternLabel(p.choice)} · nominal R${r}`),
        );
      }
    }
    choices.forEach((choice, i) => {
      const group = rows.filter(
        (row) => `${row.sampling}|${row.sampling_implementation}` === choice,
      );
      const actual = describe([
        ...new Map(group.map((row) => [row.subject, row.actual_R])).values(),
      ]);
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

  return { distribution, trend, samplingComparison };
}
