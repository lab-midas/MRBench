import { el } from "./dom.mjs";

export function validateResources(registry) {
  if (registry.schema_version !== 1) throw new Error("Unsupported resource registry");
  for (const type of ["datasets", "models"]) {
    if (!Array.isArray(registry[type])) throw new Error(`Missing ${type} registry`);
    const ids = new Set();
    for (const entry of registry[type]) {
      if (!/^[a-z0-9-]+$/.test(entry.id) || ids.has(entry.id))
        throw new Error(`Invalid or duplicate ${type} ID`);
      ids.add(entry.id);
      const structured = type === "datasets" && entry.details?.facts?.length > 0;
      if (
        !entry.name ||
        (!entry.description && !structured && !entry.sections?.length) ||
        ((type === "models" || structured) && !entry.context) ||
        !Array.isArray(entry.links)
      )
        throw new Error(`Incomplete resource: ${entry.id}`);
      if (
        entry.page_title !== undefined &&
        (typeof entry.page_title !== "string" || !entry.page_title.trim())
      )
        throw new Error(`Invalid page title: ${entry.id}`);
      if (
        entry.sections &&
        (type !== "datasets" ||
          !Array.isArray(entry.sections) ||
          !entry.sections.length ||
          entry.sections.some(
            (section) =>
              !section.title ||
              !section.description ||
              (section.facts &&
                (!Array.isArray(section.facts) ||
                  section.facts.some(
                    (row) =>
                      !row.label ||
                      !row.text ||
                      (row.status !== undefined &&
                        !["Required", "Conditional", "Optional", "Model-dependent"].includes(
                          row.status,
                        )),
                  ))) ||
              (section.references &&
                (!Array.isArray(section.references) ||
                  section.references.some(
                    (link) => !link.prefix || !link.label || !/^(https:\/\/|#\/)/.test(link.url),
                  ))) ||
              (section.note !== undefined &&
                (typeof section.note !== "string" || !section.note.trim())),
          ))
      )
        throw new Error(`Invalid resource sections: ${entry.id}`);
      if (entry.details) {
        const { facts, preview, preparation, ethics } = entry.details;
        if (
          type !== "datasets" ||
          !Array.isArray(facts) ||
          !facts.length ||
          facts.some((row) => !row.label || !row.text)
        )
          throw new Error(`Invalid dataset facts: ${entry.id}`);
        if (
          preview &&
          (!/^assets\/datasets\/[a-z0-9/-]+\.(gif|png|webp)$/.test(preview.src) ||
            !preview.alt ||
            !preview.caption ||
            !Number.isInteger(preview.width) ||
            preview.width <= 0 ||
            !Number.isInteger(preview.height) ||
            preview.height <= 0)
        )
          throw new Error(`Invalid dataset preview: ${entry.id}`);
        if (
          preparation &&
          (!Array.isArray(preparation) ||
            preparation.some((item) => {
              if (typeof item === "string") return !item.trim();
              return (
                !item?.text?.trim() ||
                !item.link?.label ||
                !/^https:\/\//.test(item.link.url) ||
                item.text.split(item.link.label).length !== 2
              );
            }))
        )
          throw new Error(`Invalid dataset preparation: ${entry.id}`);
        if (ethics !== undefined && (typeof ethics !== "string" || !ethics.trim()))
          throw new Error(`Invalid ethics statement: ${entry.id}`);
      }
      for (const link of entry.links)
        if (!link.label || !/^https:\/\//.test(link.url))
          throw new Error(`Invalid source link: ${entry.id}`);
      if (
        type === "models" &&
        (!entry.attribution ||
          !entry.paper_title ||
          !["Original paper", "Original code"].every((label) =>
            entry.links.some((link) => link.label === label),
          ))
      )
        throw new Error(`Missing model attribution: ${entry.id}`);
    }
  }
  return registry;
}

// Dataset and model entries use the same compact template, without empty sections.
export function renderResource(root, registry, route, pageHeading) {
  const type = route.page === "dataset" ? "datasets" : "models";
  const entry = registry[type].find((item) => item.id === route.id);
  if (!entry) return false;
  root.append(el("a", "← Back to overview", { href: "#/overview", class: "back-link" }));
  root.append(
    pageHeading(
      entry.page_title || entry.name,
      entry.full_name,
      route.page === "dataset" ? "Dataset" : "Model",
    ),
  );
  const content = el("div", null, { class: "resource-details" });
  if (entry.description) content.append(el("p", entry.description));
  if (entry.attribution)
    content.append(el("p", entry.attribution, { class: "resource-attribution" }));
  if (entry.details) {
    content.classList.add("dataset-details");
    const availability = el("p", null, { class: "dataset-availability" });
    availability.append(
      el("strong", "Data availability: "),
      document.createTextNode(entry.context),
    );
    content.append(availability);
    renderDatasetDetails(content, entry);
  } else if (entry.context) content.append(el("p", entry.context, { class: "resource-context" }));
  for (const [index, item] of (entry.sections || []).entries()) {
    const headingId = `resource-section-${index}`;
    const section = el("section", null, {
      class: "resource-section",
      "aria-labelledby": headingId,
    });
    section.append(el("h2", item.title, { id: headingId }), el("p", item.description));
    if (item.facts?.length)
      section.append(datasetFactsTable(item.facts, `${item.title}: inputs and requirements`));
    if (item.note) section.append(inlineText("p", item.note));
    const references = el("ul", null, { class: "resource-references" });
    for (const link of item.references || []) {
      const paragraph = el("li", link.prefix);
      const anchor = el("a", link.label, { href: link.url });
      if (link.url.startsWith("https://")) {
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
      }
      const emphasis = el("strong");
      emphasis.append(anchor);
      paragraph.append(emphasis, document.createTextNode("."));
      references.append(paragraph);
    }
    if (references.childNodes.length) section.append(references);
    content.append(section);
  }
  if (entry.links.length) {
    const links = el("div", null, { class: "resource-links", "aria-label": "Original sources" });
    for (const link of entry.links)
      links.append(
        el("a", `${link.label} ↗`, {
          href: link.url,
          target: "_blank",
          rel: "noopener noreferrer",
          ...(link.label === "Original paper" ? { title: entry.paper_title } : {}),
        }),
      );
    content.append(links);
  }
  root.append(content);
  return true;
}

function renderDatasetDetails(root, entry) {
  const { preview, facts, preparation, ethics } = entry.details;
  const layout = el("div", null, { class: "dataset-layout" });
  const media = el("div", null, { class: "dataset-media" });
  if (preview) {
    const figure = el("figure", null, { class: "dataset-preview" });
    figure.append(
      el("img", null, {
        src: preview.src,
        alt: preview.alt,
        width: preview.width,
        height: preview.height,
        decoding: "async",
      }),
      el("figcaption", preview.caption),
    );
    media.append(figure);
  }
  if (preparation?.length) {
    const section = el("section", null, { "aria-labelledby": "dataset-preparation-heading" });
    section.append(el("h2", "Input preparation", { id: "dataset-preparation-heading" }));
    const list = el("ul");
    for (const item of preparation) {
      const row = el("li");
      if (typeof item === "string") row.textContent = item;
      else {
        const [before, after] = item.text.split(item.link.label);
        row.append(
          document.createTextNode(before),
          el("a", item.link.label, {
            href: item.link.url,
            target: "_blank",
            rel: "noopener noreferrer",
          }),
          document.createTextNode(after),
        );
      }
      list.append(row);
    }
    section.append(list);
    media.append(section);
  }
  if (media.childNodes.length) layout.append(media);
  else layout.classList.add("dataset-layout-facts-only");
  const acquisition = el("section", null, { "aria-labelledby": "dataset-facts-heading" });
  acquisition.append(el("h2", "Acquisition Parameters", { id: "dataset-facts-heading" }));
  acquisition.append(datasetFactsTable(facts, "Dataset and acquisition parameters"));
  layout.append(acquisition);
  root.append(layout);
  if (ethics) root.append(el("p", ethics, { class: "dataset-ethics" }));
}

function datasetFactsTable(facts, caption) {
  const table = el("table", null, { class: "dataset-facts" });
  table.append(el("caption", caption, { class: "sr-only" }));
  const hasStatus = facts.some((fact) => fact.status);
  if (hasStatus) {
    table.classList.add("dataset-inputs");
    const head = el("thead");
    const row = el("tr");
    row.append(
      ...["Input", "Description", "Requirements"].map((label) => el("th", label, { scope: "col" })),
    );
    head.append(row);
    table.append(head);
  }
  const body = el("tbody");
  for (const fact of facts) {
    const row = el("tr");
    row.append(el("th", fact.label, { scope: "row" }), inlineText("td", fact.text));
    if (hasStatus) row.append(el("td", fact.status || "", { class: "dataset-input-status" }));
    body.append(row);
  }
  table.append(body);
  if (!hasStatus) return table;
  const scroll = el("div", null, {
    class: "dataset-inputs-scroll",
    tabindex: "0",
    role: "region",
    "aria-label": caption,
  });
  scroll.append(table);
  return scroll;
}

// Only the inline emphasis and code notation used by resource copy; no HTML is accepted.
function inlineText(tag, text) {
  const node = el(tag);
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g)) {
    if (part.startsWith("**")) node.append(el("strong", part.slice(2, -2)));
    else if (part.startsWith("`")) node.append(el("code", part.slice(1, -1)));
    else if (part.startsWith("*")) node.append(el("em", part.slice(1, -1)));
    else node.append(document.createTextNode(part));
  }
  return node;
}
