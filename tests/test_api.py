import pytest

from backend import api_routes
from backend.app import app

VALID = {
    "reading_score": 72,
    "writing_score": 74,
    "gender": "female",
    "race_ethnicity": "group C",
    "parental_level_of_education": "some college",
    "lunch": "standard",
    "test_preparation_course": "none",
}


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_valid_input_returns_score_in_range(client):
    response = client.post("/api/predict", json=VALID)
    assert response.status_code == 200
    score = response.get_json()["predicted_math_score"]
    assert 0 <= score <= 100


def test_higher_writing_score_gives_higher_prediction(client):
    low = client.post("/api/predict", json={**VALID, "writing_score": 30}).get_json()
    high = client.post("/api/predict", json={**VALID, "writing_score": 90}).get_json()
    assert high["predicted_math_score"] > low["predicted_math_score"]


def test_score_out_of_range_is_rejected(client):
    response = client.post("/api/predict", json={**VALID, "reading_score": 150})
    assert response.status_code == 422
    assert "reading_score" in " ".join(response.get_json()["details"])


def test_unknown_category_is_rejected(client):
    response = client.post("/api/predict", json={**VALID, "lunch": "pizza"})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    payload = {k: v for k, v in VALID.items() if k != "gender"}
    assert client.post("/api/predict", json=payload).status_code == 422


def test_non_json_body_is_rejected(client):
    response = client.post("/api/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_missing_model_returns_503(client, monkeypatch):
    def raise_missing():
        raise FileNotFoundError

    monkeypatch.setattr(api_routes, "get_pipeline", raise_missing)
    assert client.post("/api/predict", json=VALID).status_code == 503


def test_frontend_pages_are_served(client):
    for path in ["/", "/predict.html", "/css/style.css", "/js/app.js"]:
        assert client.get(path).status_code == 200
