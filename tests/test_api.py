from fastapi.testclient import TestClient

from app_api.main import app


client = TestClient(app)


def test_home_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Iris Classification API is running"
    }


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_setosa() -> None:
    request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == "setosa"
    assert result["prediction_id"] == 0
    assert "probabilities" in result
    assert "setosa" in result["probabilities"]


def test_predict_validation_error() -> None:
    invalid_request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }

    response = client.post(
        "/predict",
        json=invalid_request_data,
    )

    assert response.status_code == 422