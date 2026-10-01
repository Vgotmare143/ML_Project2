from flask import Blueprint, jsonify, request

from ml_pipeline.pipelines.predict_pipeline import PredictPipeline, validate

api = Blueprint("api", __name__)
_pipeline = None


def get_pipeline() -> PredictPipeline:
    """Load the model lazily so the server can start before training has run."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictPipeline()
    return _pipeline


@api.get("/health")
def health():
    return jsonify(status="ok")


@api.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be JSON."), 400

    errors = validate(payload)
    if errors:
        return jsonify(error="Invalid input.", details=errors), 422

    try:
        score = get_pipeline().predict(payload)
    except FileNotFoundError:
        return jsonify(error="Model not found. Run the training pipeline first."), 503
    return jsonify(predicted_math_score=score)
