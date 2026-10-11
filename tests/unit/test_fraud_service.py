import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient

import services.fraud.main as fraud_main
from common.config import settings
from services.fraud.features import FEATURE_NAMES
from services.fraud.model import FraudModel


@pytest.fixture
def client(monkeypatch, fraud_model_prefix):
    monkeypatch.setattr(fraud_main, "_redis", fakeredis.FakeRedis(decode_responses=True))
    monkeypatch.setattr(settings, "fraud_model_path", str(fraud_model_prefix))
    with TestClient(fraud_main.app) as c:
        yield c


def payment(i, **overrides):
    body = {
        "event_id": f"evt_{i}",
        "card_fingerprint": f"fp_{i}",
        "bin": "424242",
        "ip": f"10.0.{i // 250}.{i % 250}",
        "email": f"customer{i}@example.com",
        "amount_cents": 4200,
        "merchant_id": "merchant_01",
    }
    return body | overrides


def outcome_for(body, declined):
    keys = ("event_id", "card_fingerprint", "bin", "ip", "email")
    return {k: body[k] for k in keys} | {"declined": declined}


def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["model"] == "gbm"
    assert 0 < body["threshold"] <= 1


def test_normal_payment_is_allowed(client):
    resp = client.post("/v1/score", json=payment(1))
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "allow"
    assert set(body["features"]) == set(FEATURE_NAMES)


def test_card_testing_burst_gets_blocked(client):
    decisions = []
    for i in range(40):
        body = payment(i, ip="66.66.66.66", email="tester@example.com", amount_cents=100)
        decisions.append(client.post("/v1/score", json=body).json()["decision"])
        client.post("/v1/outcome", json=outcome_for(body, declined=True))
    assert decisions[0] == "allow"
    assert decisions[-1] == "block"


def test_outcome_feeds_decline_rate(client):
    first = payment(1)
    client.post("/v1/score", json=first)
    client.post("/v1/outcome", json=outcome_for(first, declined=True))
    second = payment(2, ip=first["ip"])
    assert client.post("/v1/score", json=second).json()["features"]["decline_rate_ip_5m"] == 1.0


@pytest.mark.parametrize(
    ("risk", "expected"), [(0.0, "allow"), (0.5, "review"), (0.89, "review"), (0.9, "block")]
)
def test_decision_tiers(client, monkeypatch, risk, expected):
    monkeypatch.setattr(settings, "fraud_review_threshold", 0.5)
    stub = FraudModel(model=None, meta={"model": "stub", "threshold": 0.9, "features": []})
    monkeypatch.setattr(stub, "predict", lambda features, amount: risk)
    client.app.state.model = stub
    assert client.post("/v1/score", json=payment(1)).json()["decision"] == expected


def test_missing_artifact_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="ml.train"):
        FraudModel.load(tmp_path / "nope")


class _DownRedis(fakeredis.FakeRedis):
    def pipeline(self, *args, **kwargs):
        raise redis.ConnectionError("connection refused")

    def zscore(self, *args, **kwargs):
        raise redis.ConnectionError("connection refused")


def test_redis_down_returns_503(client, monkeypatch):
    monkeypatch.setattr(fraud_main, "_redis", _DownRedis(decode_responses=True))
    resp = client.post("/v1/score", json=payment(1))
    assert resp.status_code == 503
    assert "feature store unavailable" in resp.json()["detail"]
    outcome = client.post("/v1/outcome", json=outcome_for(payment(1), declined=True))
    assert outcome.status_code == 503


@pytest.mark.parametrize("bad", [{"bin": "42424"}, {"amount_cents": 0}, {"ip": None}])
def test_invalid_request_is_rejected(client, bad):
    assert client.post("/v1/score", json=payment(1) | bad).status_code == 422
