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
      if (!entry.name || !entry.description || !entry.context || !Array.isArray(entry.links))
        throw new Error(`Incomplete resource: ${entry.id}`);
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
    pageHeading(entry.name, entry.full_name, route.page === "dataset" ? "Dataset" : "Model"),
  );
  const content = el("div", null, { class: "resource-details" });
  content.append(el("p", entry.description));
  if (entry.attribution)
    content.append(el("p", entry.attribution, { class: "resource-attribution" }));
  content.append(el("p", entry.context, { class: "resource-context" }));
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
