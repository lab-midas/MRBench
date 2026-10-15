export const acceleration = Object.freeze({
  title: "Across accelerations",
  container: "trend",
  caption:
    "Mean ± sample SD at each nominal acceleration, for the selected sampling pattern. The guide marks the condition shown on the left.",
  draw(charts) {
    charts.trend();
  },
});
