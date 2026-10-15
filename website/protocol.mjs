import { el } from "./dom.mjs";

// Presentation only: these equations describe the Python evaluator, never calculate metrics.
// Verified source/function references are recorded in docs/protocol-verification.md.
export const PROTOCOL = Object.freeze([
  {
    id: "normalization",
    title: "Magnitude and subject normalization",
    paragraphs: [
      "Each reference and reconstruction is converted to magnitude. For each source U, one scale is computed over its complete subject volume, including every slice, time frame and spatial pixel. Reference, candidate and zero-filled sources are normalized independently.",
      "P₉₉.₅ is the 99.5th percentile of that source's complete magnitude volume. No framewise normalization, reference-fitted gain or clipping is applied. Values above one remain in the metric inputs. Below, r and x denote normalized reference and reconstruction frames, respectively.",
    ],
    equations: [
      {
        label: "a sub U equals the 99.5th percentile of the magnitude of U",
        math: '<msub><mi>a</mi><mi>U</mi></msub><mo>=</mo><msub><mi mathvariant="normal">P</mi><mn>99.5</mn></msub><mo>(</mo><mo>|</mo><mi>U</mi><mo>|</mo><mo>)</mo>',
      },
      {
        label: "Normalized U equals magnitude U divided by its own a sub U",
        math: "<mover><mi>U</mi><mo>~</mo></mover><mo>=</mo><mfrac><mrow><mo>|</mo><mi>U</mi><mo>|</mo></mrow><msub><mi>a</mi><mi>U</mi></msub></mfrac>",
      },
    ],
  },
  {
    id: "nrmse",
    title: "NRMSE",
    direction: "Lower is better",
    paragraphs: [
      "Normalized root mean squared error is the L2 reconstruction error divided by the L2 norm of the normalized reference frame. It is computed separately for each 2-D slice/phase frame.",
    ],
    equations: [
      {
        label: "NRMSE equals L2 norm of x minus r divided by L2 norm of r",
        math: '<mi mathvariant="normal">NRMSE</mi><mo>=</mo><mfrac><mrow><mo>‖</mo><mi>x</mi><mo>−</mo><mi>r</mi><msub><mo>‖</mo><mn>2</mn></msub></mrow><mrow><mo>‖</mo><mi>r</mi><msub><mo>‖</mo><mn>2</mn></msub></mrow></mfrac>',
      },
    ],
  },
  {
    id: "psnr",
    title: "PSNR",
    direction: "Higher is better · dB",
    paragraphs: [
      "Mean squared error averages the squared difference over the N spatial pixels in a frame. Peak signal-to-noise ratio uses the fixed data range L = 1, including when normalized values exceed one. The range is not estimated from that frame or candidate.",
    ],
    equations: [
      {
        label: "MSE equals one over N times the sum of squared pixel differences",
        math: '<mi mathvariant="normal">MSE</mi><mo>=</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><msup><mrow><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>−</mo><msub><mi>r</mi><mi>i</mi></msub><mo>)</mo></mrow><mn>2</mn></msup>',
      },
      {
        label: "PSNR equals ten log base ten of L squared over MSE, with L equal to one",
        math: '<mi mathvariant="normal">PSNR</mi><mo>=</mo><mn>10</mn><msub><mi mathvariant="normal">log</mi><mn>10</mn></msub><mo>(</mo><mfrac><msup><mi>L</mi><mn>2</mn></msup><mi mathvariant="normal">MSE</mi></mfrac><mo>)</mo><mo>,</mo><mspace width="0.5em"/><mi>L</mi><mo>=</mo><mn>1</mn>',
      },
    ],
  },
  {
    id: "ssim",
    title: "SSIM",
    direction: "Higher is better",
    paragraphs: [
      "Structural similarity combines local means, variances and covariance. The expression below is evaluated at each valid window center p, then averaged to obtain the frame's SSIM.",
      "The evaluator uses scikit-image structural_similarity with a uniform 7 × 7 window, no Gaussian weighting and data range L = 1. Library defaults use sample covariance, K₁ = 0.01 and K₂ = 0.03, so C₁ = (K₁L)² and C₂ = (K₂L)². The frame mean excludes the three-pixel border where a full window does not fit.",
    ],
    equations: [
      {
        label: "Local SSIM S of p equals the luminance factor times the covariance factor",
        parts: [
          "<mi>S</mi><mo>(</mo><mi>p</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mn>2</mn><msub><mi>μ</mi><mi>r</mi></msub><msub><mi>μ</mi><mi>x</mi></msub><mo>+</mo><msub><mi>C</mi><mn>1</mn></msub></mrow><mrow><msubsup><mi>μ</mi><mi>r</mi><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>μ</mi><mi>x</mi><mn>2</mn></msubsup><mo>+</mo><msub><mi>C</mi><mn>1</mn></msub></mrow></mfrac>",
          "<mo>×</mo><mfrac><mrow><mn>2</mn><msub><mi>σ</mi><mrow><mi>r</mi><mi>x</mi></mrow></msub><mo>+</mo><msub><mi>C</mi><mn>2</mn></msub></mrow><mrow><msubsup><mi>σ</mi><mi>r</mi><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>σ</mi><mi>x</mi><mn>2</mn></msubsup><mo>+</mo><msub><mi>C</mi><mn>2</mn></msub></mrow></mfrac>",
        ],
      },
      {
        label: "Frame SSIM is the arithmetic mean of local S of p over valid window centers Omega",
        math: '<mi mathvariant="normal">SSIM</mi><mo>=</mo><mfrac><mn>1</mn><mrow><mo>|</mo><mi>Ω</mi><mo>|</mo></mrow></mfrac><munder><mo>∑</mo><mrow><mi>p</mi><mo>∈</mo><mi>Ω</mi></mrow></munder><mi>S</mi><mo>(</mo><mi>p</mi><mo>)</mo>',
      },
    ],
  },
]);

function equation({ math, parts, label }) {
  const wrapper = el("div", null, { class: "equation", role: "math", "aria-label": label });
  for (const markup of parts || [math]) {
    // The markup is a static, authored MathML constant, never registry/user input.
    const template = document.createElement("template");
    template.innerHTML = `<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-hidden="true"><mrow>${markup}</mrow></math>`;
    wrapper.append(template.content);
  }
  return wrapper;
}
function article(title, paragraphs, className = "") {
  const node = el("article", null, { class: className });
  node.append(el("h2", title), ...paragraphs.map((text) => el("p", text)));
  return node;
}

export function renderProtocol(root, pageHeading) {
  root.append(
    pageHeading("Evaluation protocol", "How reconstruction quality is measured and compared."),
  );
  const metrics = el("div", null, { class: "protocol-metrics" });
  for (const metric of PROTOCOL) {
    const card = el("article", null, {
      class: `metric-card metric-${metric.id}`,
      id: `protocol-${metric.id}`,
    });
    const heading = el("div", null, { class: "metric-heading" });
    heading.append(el("h2", metric.title));
    if (metric.direction)
      heading.append(el("span", metric.direction, { class: "metric-direction" }));
    card.append(heading, el("p", metric.paragraphs[0]));
    const equations = el("div", null, { class: "equations" });
    equations.append(...metric.equations.map(equation));
    card.append(equations, ...metric.paragraphs.slice(1).map((text) => el("p", text)));
    metrics.append(card);
  }
  root.append(metrics);
  const aggregation = article(
    "From frames to subjects to a cohort",
    [
      "For subject i, each metric is averaged equally over all Sᵢ × Tᵢ slice/phase frames. Subjects then receive equal weight in the cohort mean; frames from different subjects are not pooled together.",
      "For a cohort of n subjects, plots report the mean subject endpoint μ and sample standard deviation s. SD describes between-subject variation, not a confidence interval. Boxplots show the median and interquartile range, with whiskers spanning the observed subject range.",
    ],
    "protocol-aggregation",
  );
  aggregation.append(
    equation({
      label:
        "Subject metric m sub i equals one over S sub i T sub i times the sum of all slice phase frame metrics",
      math: "<msub><mi>m</mi><mi>i</mi></msub><mo>=</mo><mfrac><mn>1</mn><mrow><msub><mi>S</mi><mi>i</mi></msub><msub><mi>T</mi><mi>i</mi></msub></mrow></mfrac><munder><mo>∑</mo><mrow><mi>s</mi><mo>,</mo><mi>t</mi></mrow></munder><msub><mi>m</mi><mrow><mi>i</mi><mo>,</mo><mi>s</mi><mo>,</mo><mi>t</mi></mrow></msub>",
    }),
    equation({
      label: "Cohort mean mu equals one over n times the sum of subject metrics",
      math: "<mi>μ</mi><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msub><mi>m</mi><mi>i</mi></msub>",
    }),
    equation({
      label:
        "Sample SD s equals the square root of the sum of squared subject deviations divided by n minus one",
      math: "<mi>s</mi><mo>=</mo><msqrt><mfrac><mrow><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msup><mrow><mo>(</mo><msub><mi>m</mi><mi>i</mi></msub><mo>−</mo><mi>μ</mi><mo>)</mo></mrow><mn>2</mn></msup></mrow><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></mfrac></msqrt>",
    }),
  );
  root.append(aggregation);
  const context = el("div", null, { class: "protocol-grid" });
  context.append(
    article("Nominal and actual acceleration", [
      "Nominal R is a sampler setting. Actual R is the number of complete-grid locations divided by the number of acquired locations in the full realized binary mask, including ACS. It can differ across sampling patterns at the same nominal R.",
      "The current in-house cardiac cine study constructs multi-coil k-space from prepared complex reference images and sensitivity maps. Its pseudo-radial masks are rasterized Cartesian support, rather than non-Cartesian NUFFT acquisitions.",
    ]),
    article("Matched measurements and model conditioning", [
      "Models share the same physical mask for each subject and condition, verified by mask hashes. Subject identifiers, acquisition settings and reference construction remain matched. Model-specific normalization and conditioning are retained.",
      "RAM receives sensitivity maps and a multi-coil operator. SDUM uses temporal windows and internal map estimation. CardioMM uses framewise reconstruction, internal maps and acquisition metadata. Physical sampling and model-conditioning labels are recorded separately.",
    ]),
    article("Amplitude and interpretation", [
      "Independent normalization removes one positive global output gain. Raw-amplitude errors and scale ratios remain separate diagnostics when interpreting intensity-scaling robustness. For the current in-house study, the reference is the canonical magnitude image; it is not replaced by an RSS target.",
      "Different datasets and metric protocols require separately audited tables. Runtime fields are available in the downloaded data but are not ranked because timing scopes and hardware are not harmonized.",
    ]),
    article("Validity and coverage", [
      "Nonfinite inputs, nonpositive normalization scales, incompatible shapes and zero-reference frames are invalid conditions. They are not repaired with an epsilon or replaced by zero scores. SSIM requires both spatial dimensions to contain at least seven pixels.",
      "An exact zero frame MSE gives infinite PSNR. The published cohort table admits only finite, passed endpoints and verified matching masks; failures and missing results require separate coverage reporting.",
    ]),
  );
  root.append(context);
}
