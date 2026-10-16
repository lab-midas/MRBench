import { el } from "./dom.mjs";
import { FRAMEWORK, hotspotURL, hotspotStyle } from "./framework-registry.mjs";

export function mountFramework(root) {
  const section = el("section", null, {
    class: "framework-section",
    "aria-labelledby": "framework-heading",
  });
  section.append(el("h2", "Framework overview", { id: "framework-heading" }));
  const figure = el("figure", null, { class: "framework-figure" });
  const scroll = el("div", null, {
    class: "framework-scroll",
    role: "region",
    "aria-label": "Interactive framework illustration",
    tabindex: "0",
    "aria-describedby": "framework-instructions",
  });
  const canvas = el("div", null, { class: "framework-canvas" });
  canvas.append(
    el("img", null, {
      src: FRAMEWORK.image.src,
      alt: FRAMEWORK.image.alt,
      class: "framework-image",
      width: FRAMEWORK.image.width,
      height: FRAMEWORK.image.height,
      decoding: "async",
      draggable: "false",
    }),
  );
  const overlay = el("div", null, { class: "framework-overlay" });
  for (const hotspot of FRAMEWORK.hotspots.filter((item) => item.enabled)) {
    const link = el("a", null, {
      href: hotspotURL(hotspot),
      class: "framework-hotspot",
      "data-hotspot": hotspot.id,
      "aria-label": hotspot.label,
      title: hotspot.label,
      style: hotspotStyle(hotspot.rect),
    });
    link.append(el("span", hotspot.label, { class: "sr-only" }));
    overlay.append(link);
  }
  canvas.append(overlay);
  scroll.append(canvas);
  figure.append(scroll);
  const caption = el(
    "figcaption",
    "Select a dataset, experiment category, model or the metrics panel to explore.",
    { id: "framework-instructions" },
  );
  caption.append(
    el("span", " On small screens, swipe horizontally to view the complete illustration.", {
      class: "framework-mobile-note",
    }),
  );
  figure.append(caption);
  section.append(figure);
  root.append(section);
}
