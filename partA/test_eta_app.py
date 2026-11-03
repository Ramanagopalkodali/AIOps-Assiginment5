"""Contract tests for the SwiftParcel ETA service (assignment Part A.2).

Run from the repository root: python -m pytest -q partA/test_eta_app.py
"""

import re
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from partA import eta_app


URL = "/v1/models/eta:predict"
ROW = {
    "distance_km": 12.5,
    "n_stops": 3,
    "hour": 11,
    "weekday": 2,
    "traffic_index": 0.35,
    "rain": 0,
    "weight_kg": 4.0,
}


@pytest.fixture
def client():
    with TestClient(eta_app.app) as test_client:
        yield test_client


def post(client, rows, headers=None):
    return client.post(URL, json={"instances": rows}, headers=headers or {})


def assert_validation(response, field):
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_failed"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert any(field in detail["field"] for detail in body["details"])
    assert all("msg" in detail for detail in body["details"])


def test_single_prediction_and_version(client):
    response = post(client, [ROW])
    assert response.status_code == 200
    body = response.json()
    expected = eta_app.model.predict(pd.DataFrame([ROW])[eta_app.feature_columns])[0]
    assert body["predictions"] == [round(float(expected), 1)]
    assert body["model_version"] == eta_app.model_version
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_batch_preserves_order(client):
    rows = [ROW, {**ROW, "distance_km": 72.0, "n_stops": 12},
            {**ROW, "distance_km": 5.0, "n_stops": 1}]
    response = post(client, rows)
    assert response.status_code == 200
    expected = [round(float(x), 1) for x in
                eta_app.model.predict(pd.DataFrame(rows)[eta_app.feature_columns])]
    assert len(set(expected)) == 3
    assert response.json()["predictions"] == expected


def test_distance_below_minimum(client):
    assert_validation(post(client, [{**ROW, "distance_km": 0.09}]), "distance_km")


def test_hour_outside_range(client):
    assert_validation(post(client, [{**ROW, "hour": 24}]), "hour")


def test_invalid_rain_value(client):
    assert_validation(post(client, [{**ROW, "rain": 2}]), "rain")


def test_missing_feature(client):
    incomplete = {key: value for key, value in ROW.items() if key != "weight_kg"}
    assert_validation(post(client, [incomplete]), "weight_kg")


def test_infeasible_combination_and_boundary(client):
    assert post(client, [{**ROW, "distance_km": 150, "n_stops": 21}]).status_code == 200
    response = post(client, [{**ROW, "distance_km": 150.01, "n_stops": 21}])
    assert response.status_code == 422
    assert "infeasible" in str(response.json()["details"]).lower()


def test_empty_batch_rejected(client):
    assert_validation(post(client, []), "instances")


def test_one_row_batch_accepted(client):
    response = post(client, [ROW])
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_500_rows_accepted_501_rejected(client):
    accepted = post(client, [ROW] * 500)
    assert accepted.status_code == 200
    assert len(accepted.json()["predictions"]) == 500
    assert_validation(post(client, [ROW] * 501), "instances")


def test_request_id_echo_on_success_and_error(client):
    for rows, status in [([ROW], 200), ([{**ROW, "distance_km": -1}], 422)]:
        response = post(client, rows, {"X-Request-ID": "customer-request-42"})
        assert response.status_code == status
        assert response.headers["X-Request-ID"] == "customer-request-42"
        assert response.json()["request_id"] == "customer-request-42"


def test_generated_request_id_on_success_and_error(client):
    for rows, status in [([ROW], 200), ([], 422)]:
        response = post(client, rows)
        assert response.status_code == status
        generated = response.headers["X-Request-ID"]
        assert str(uuid.UUID(generated)) == generated
        assert response.json()["request_id"] == generated


def test_health_reports_loaded_version(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == eta_app.model_version
    assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_metrics_counts_requests_but_not_metrics_endpoint(client):
    def count(text):
        match = re.search(r'^eta_requests_total\{path="/v1/models/eta:predict",status="200"\} (\S+)$',
                          text, re.MULTILINE)
        return float(match.group(1)) if match else 0.0

    before = client.get("/metrics")
    assert before.status_code == 200
    assert "text/plain" in before.headers["content-type"]
    old_count = count(before.text)
    assert post(client, [ROW]).status_code == 200
    after = client.get("/metrics")
    assert count(after.text) == old_count + 1
    assert 'path="/metrics"' not in after.text
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in after.text
    assert 'eta_request_latency_seconds_sum' in after.text
    assert 'eta_request_latency_seconds_count' in after.text


def test_inference_failure_is_generic_500(client, monkeypatch):
    class BrokenModel:
        def predict(self, frame):
            raise RuntimeError("SECRET INTERNAL MODEL ERROR")

    monkeypatch.setattr(eta_app, "model", BrokenModel())
    response = post(client, [ROW], {"X-Request-ID": "failed-request"})
    assert response.status_code == 500
    assert "SECRET" not in response.text
    assert response.headers["X-Request-ID"] == "failed-request"
    assert response.json()["request_id"] == "failed-request"
