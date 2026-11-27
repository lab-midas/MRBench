export const sampling = Object.freeze({
  title: "Across sampling patterns",
  container: "sampling-comparison",
  defaultAcceleration: 16,
  caption: "Mean ± SD for each sampling pattern at the selected nominal acceleration.",
  draw(charts) {
    charts.samplingComparison();
  },
});
