from pathlib import Path

import joblib
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"

NUMERIC_FIELDS = ["reading_score", "writing_score"]
CATEGORICAL_OPTIONS = {
    "gender": ["female", "male"],
    "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
    "parental_level_of_education": [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree",
    ],
    "lunch": ["standard", "free/reduced"],
    "test_preparation_course": ["none", "completed"],
}
FEATURES = NUMERIC_FIELDS + list(CATEGORICAL_OPTIONS)


class PredictPipeline:
    """Loads the saved preprocessor and model once, then scores new students."""

    def __init__(self):
        self.preprocessor = joblib.load(ARTIFACTS / "preprocessor.pkl")
        self.model = joblib.load(ARTIFACTS / "model.pkl")

    def predict(self, payload: dict) -> float:
        frame = pd.DataFrame([payload], columns=FEATURES)
        features = self.preprocessor.transform(frame)
        score = float(self.model.predict(features)[0])
        return round(min(max(score, 0.0), 100.0), 1)  # keep within a valid 0-100 score


def validate(payload: dict) -> list[str]:
    """Return a list of human-readable errors (empty list means valid)."""
    errors = []
    for field in NUMERIC_FIELDS:
        try:
            value = float(payload.get(field))
            if not 0 <= value <= 100:
                errors.append(f"{field} must be between 0 and 100.")
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number.")
    for field, options in CATEGORICAL_OPTIONS.items():
        if payload.get(field) not in options:
            errors.append(f"{field} must be one of: {', '.join(options)}.")
    return errors
