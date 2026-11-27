import { el } from "./dom.mjs";
import { modelColor, modelLabel } from "./models.mjs";

// A collection holds comparable image panels for one or more experiment conditions.
export function hasPublishedMedia(collections) {
  return collections.some((c) =>
    c.conditions?.some((condition) => condition.panels?.some((panel) => panel.src)),
  );
}
export function validateMedia(collections) {
  for (const collection of collections) {
    if (!collection.id || !collection.title || !collection.conditions?.length)
      throw new Error("Incomplete image collection");
    if (collection.playback) {
      const { width, height, frames, frameMs } = collection.playback;
      if (
        ![width, height, frames, frameMs].every((n) => Number.isInteger(n) && n > 0) ||
        frames < 2
      )
        throw new Error("Invalid cine playback settings");
    }
    const ids = new Set();
    const selections = new Set();
    for (const condition of collection.conditions) {
      if (!condition.id || ids.has(condition.id) || !condition.label || !condition.panels?.length)
        throw new Error("Incomplete image condition");
      ids.add(condition.id);
      if (collection.followFilters) {
        const key = JSON.stringify([condition.sampling, condition.acceleration]);
        if (
          !condition.sampling ||
          !Number.isFinite(condition.acceleration) ||
          condition.acceleration <= 0 ||
          selections.has(key)
        )
          throw new Error("Invalid or duplicate image filter condition");
        selections.add(key);
      }
      for (const panel of condition.panels) {
        if (
          !panel.alt ||
          !panel.label ||
          !["reference", "reconstruction", "error", "mask", "input", "zero-filled"].includes(
            panel.role,
          )
        )
          throw new Error("Incomplete image panel");
        if (collection.followFilters && panel.role === "reference" && panel.src === null) continue;
        if (!/^assets\/experiments\/[a-z0-9/_-]+\.(png|gif)$/i.test(panel.src))
          throw new Error("Use experiment-local pre-rendered PNG or GIF assets");
        if (collection.playback && !panel.src.endsWith(".png"))
          throw new Error("Pausable cine panels require PNG frame strips");
      }
    }
  }
  return collections;
}
export function mountMedia(root, collections, heading, id, { showEmpty = false } = {}) {
  if (!showEmpty && !hasPublishedMedia(collections)) return;
  validateMedia(collections);
  const section = el("section", null, { class: "media-section" });
  const sharedFilters = [];
  if (id) section.id = id;
  section.append(el("h2", heading));
  for (const collection of collections) {
    const article = el("article", null, { class: "media-collection" });
    const title = el("h3", collection.title);
    const grid = el("div", null, { class: "media-grid" });
    const playback = collection.playback;
    let playbackButton;
    if (playback) {
      const { width, height, frames, frameMs } = playback;
      grid.classList.add("cine-grid");
      grid.style.setProperty("--cine-aspect", `${width} / ${height}`);
      grid.style.setProperty("--cine-frames", frames);
      grid.style.setProperty("--cine-duration", `${frames * frameMs}ms`);
      const toolbar = el("div", null, { class: "cine-toolbar" });
      playbackButton = el("button", null, { type: "button", class: "cine-playback" });
      playbackButton.hidden = true;
      const setPaused = (paused) => {
        grid.classList.toggle("cine-paused", paused);
        playbackButton.textContent = paused ? "Play all" : "Pause all";
      };
      setPaused(window.matchMedia("(prefers-reduced-motion: reduce)").matches);
      playbackButton.addEventListener("click", () => {
        setPaused(!grid.classList.contains("cine-paused"));
      });
      toolbar.append(title, playbackButton);
      article.append(toolbar);
    } else article.append(title);

    const draw = (condition) => {
      if (playbackButton) playbackButton.hidden = !condition.panels.some((panel) => panel.src);
      grid.replaceChildren(
        ...condition.panels.map((panel) => {
          const figure = el("figure", null, {
            class: "media-panel",
            ...(playback ? { "data-model": panel.model ?? panel.role } : {}),
          });
          if (panel.model && !playback) figure.style.borderTopColor = modelColor(panel.model);
          const variantRows = panel.variant
            ? [{ model: panel.model, variant_or_implementation: panel.variant }]
            : [];
          const caption = el(
            "figcaption",
            `${panel.model && !collection.followFilters ? modelLabel(panel.model, variantRows) + " · " : ""}${panel.label}`,
          );
          let image;
          if (!panel.src) {
            image = el("p", "Reference unavailable", { class: "media-unavailable" });
          } else if (playback) {
            image = el("div", null, { class: "cine-frame", role: "img", "aria-label": panel.alt });
            image.style.backgroundImage = `url("${panel.src}")`;
          } else {
            image = el("img", null, { src: panel.src, alt: panel.alt, loading: "lazy" });
          }
          if (playback) figure.append(caption, image);
          else figure.append(image, caption);
          return figure;
        }),
      );
    };
    if (collection.followFilters) {
      grid.classList.add("media-grid-linked");
      const status = el("p", "Select a sampling pattern and nominal acceleration above.", {
        class: "media-status",
        role: "status",
      });
      article.append(status);
      let previous;
      sharedFilters.push(({ sampling, R }) => {
        const key = JSON.stringify([sampling, R]);
        if (key === previous) return;
        previous = key;
        const condition = collection.conditions.find(
          (entry) => entry.sampling === sampling && entry.acceleration === Number(R),
        );
        status.textContent = condition
          ? condition.label
          : "Reconstruction images are unavailable for the selected condition.";
        if (condition) draw(condition);
        else {
          grid.replaceChildren();
          if (playbackButton) playbackButton.hidden = true;
        }
      });
    } else if (collection.conditions.length > 1) {
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
    if (!collection.followFilters) draw(collection.conditions[0]);
    if (collection.caption) article.append(el("p", collection.caption, { class: "chart-caption" }));
    section.append(article);
  }
  root.append(section);
  if (sharedFilters.length) return (selection) => sharedFilters.forEach((update) => update(selection));
}
