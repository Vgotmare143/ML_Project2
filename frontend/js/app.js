const form = document.getElementById("predict-form");
const button = document.getElementById("submit-btn");
const result = document.getElementById("result");
const scoreEl = document.getElementById("score");
const bandEl = document.getElementById("band");
const unitEl = document.getElementById("unit");
const rangeEl = document.getElementById("range");

// Average test error of the trained model (MAE), in score points.
const TYPICAL_ERROR = 4;

function describe(score) {
  if (score >= 80) return "High predicted score";
  if (score >= 60) return "Moderate predicted score";
  if (score >= 40) return "Below-average predicted score";
  return "Low predicted score";
}

function showResult(text, caption, isError = false, score = null) {
  result.hidden = false;
  result.classList.toggle("error", isError);
  scoreEl.textContent = text;
  bandEl.textContent = caption;
  // Unit and error range only apply to a real prediction
  unitEl.textContent = isError ? "" : "out of 100, predicted math score";
  rangeEl.textContent = isError
    ? ""
    : `Typical error: about ±${TYPICAL_ERROR} points (${Math.max(0, score - TYPICAL_ERROR).toFixed(0)} to ${Math.min(100, score + TYPICAL_ERROR).toFixed(0)})`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = Object.fromEntries(new FormData(form).entries());
  payload.reading_score = Number(payload.reading_score);
  payload.writing_score = Number(payload.writing_score);

  button.disabled = true;
  button.textContent = "Predicting...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    if (!response.ok) {
      const detail = (data.details || [data.error]).join(" ");
      showResult("Could not predict", detail, true);
      return;
    }
    const score = data.predicted_math_score;
    showResult(score.toFixed(1), describe(score), false, score);
  } catch (err) {
    showResult("Server unreachable", "Check that the backend is running and try again.", true);
  } finally {
    button.disabled = false;
    button.textContent = "Predict score";
  }
});