"""API compatibility tests."""

import pytest
from fastapi.testclient import TestClient

from openfugu.api.app import create_app
from openfugu.config import AppConfig, WorkerConfig


@pytest.fixture
def client():
    config = AppConfig(
        workers=[
            WorkerConfig(
                id=0,
                name="mock-worker",
                provider="openai",
                model="gpt-4o-mini",
                capabilities=["chat"],
            )
        ]
    )
    app = create_app(config)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "openfugu" in ids
    assert "openfugu-ultra" in ids


def test_rewards_soft_targets():
    from openfugu.training.rewards import router_soft_targets

    targets = router_soft_targets([0.9, 0.6, 0.3], temperature=1.0)
    assert abs(sum(targets) - 1.0) < 1e-6
    assert targets[0] > targets[1] > targets[2]
