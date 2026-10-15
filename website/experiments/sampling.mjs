export const sampling = Object.freeze({
  title: "Across sampling patterns",
  container: "sampling-comparison",
  caption:
    "Grouped means ± sample SD at the selected nominal R. Patterns are categorical; actual R can differ between patterns.",
  draw(charts) {
    charts.samplingComparison();
  },
});
