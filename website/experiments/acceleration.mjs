export const acceleration = Object.freeze({
  title: "Across accelerations",
  container: "trend",
  caption: "Mean ± SD at each nominal acceleration, for the selected sampling pattern.",
  draw(charts) {
    charts.trend();
  },
});
