import { el } from "./dom.mjs";
import { modelColor, modelLabel } from "./models.mjs";

// A collection holds selectable conditions, each with side-by-side PNG panels.
// Nothing is rendered for an empty collection: there are no placeholder figures.
export function hasPublishedMedia(collections) {
  return collections.some((c) => c.conditions?.some((condition) => condition.panels?.length));
}
export function validateMedia(collections) {
  for (const collection of collections) {
    if (!collection.id || !collection.title || !collection.conditions?.length)
      throw new Error("Incomplete image collection");
    const ids = new Set();
    for (const condition of collection.conditions) {
      if (!condition.id || ids.has(condition.id) || !condition.label || !condition.panels?.length)
        throw new Error("Incomplete image condition");
      ids.add(condition.id);
      for (const panel of condition.panels) {
        if (
          !panel.alt ||
          !panel.label ||
          !["reference", "reconstruction", "error", "mask", "input", "zero-filled"].includes(
            panel.role,
          )
        )
          throw new Error("Incomplete image panel");
        if (!/^assets\/experiments\/[a-z0-9/_-]+\.png$/i.test(panel.src))
          throw new Error("Use experiment-local pre-rendered PNG assets");
      }
    }
  }
  return collections;
}
export function mountMedia(root, collections, heading, id) {
  if (!hasPublishedMedia(collections)) return;
  validateMedia(collections);
  const section = el("section", null, { class: "media-section" });
  if (id) section.id = id;
  section.append(el("h2", heading));
  for (const collection of collections) {
    const article = el("article", null, { class: "media-collection" });
    article.append(el("h3", collection.title));
    const grid = el("div", null, { class: "media-grid" });
    const draw = (condition) => {
      grid.replaceChildren(
        ...condition.panels.map((panel) => {
          const figure = el("figure", null, { class: "media-panel" });
          if (panel.model) figure.style.borderTopColor = modelColor(panel.model);
          const variantRows = panel.variant
            ? [{ model: panel.model, variant_or_implementation: panel.variant }]
            : [];
          figure.append(
            el("img", null, { src: panel.src, alt: panel.alt, loading: "lazy" }),
            el(
              "figcaption",
              `${panel.model ? modelLabel(panel.model, variantRows) + " · " : ""}${panel.label}`,
            ),
          );
          return figure;
        }),
      );
    };
    if (collection.conditions.length > 1) {
      const label = el("label", "Condition", { class: "media-filter" });
      const select = el("select");
      for (const condition of collection.conditions)
        select.append(el("option", condition.label, { value: condition.id }));
      select.addEventListener("change", () =>
        draw(collection.conditions.find((c) => c.id === select.value)),
      );
      label.append(select);
      article.append(label);
    }
    article.append(grid);
    if (collection.caption) article.append(el("p", collection.caption, { class: "chart-caption" }));
    draw(collection.conditions[0]);
    section.append(article);
  }
  root.append(section);
}
