"""Test FastAPI endpoints for ResQ-AI."""

import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_config():
    response = client.get("/config")
    assert response.status_code == 200
    assert "uq_methods" in response.json()

def test_predict():
    response = client.post("/predict", json={"uq_method": "ensemble", "use_demo": True})
    assert response.status_code == 200
    data = response.json()
    assert "probability_map" in data
    assert "uncertainty_map" in data
    assert "confidence_interval" in data

def test_impact():
    response = client.post("/impact", json={
        "prediction_id": "test_1",
        "population_data": {"count": 1000},
        "building_data": {"count": 200}
    })
    assert response.status_code == 200
    assert "impact_distribution" in response.json()

def test_allocate():
    response = client.post("/allocate", json={
        "impact_scenarios": [{"zone_1": 50}],
        "depots": [{"id": "depot_1", "capacity": 100}],
        "method": "cvar",
        "alpha": 0.95,
        "num_scenarios": 10
    })
    assert response.status_code == 200
    assert "allocation_plan" in response.json()
