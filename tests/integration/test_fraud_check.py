import random
import uuid

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

import services.fraud.main as fraud_main
import services.gateway.fraud as gateway_fraud
import services.gateway.routes as gateway_routes
from common.config import settings
from common.http import UpstreamError
from common.luhn import complete_luhn
from services.fraud.model import FraudModel
from services.gateway.db import FraudCheck, get_session

AUTH_HEADERS = {"Authorization": "Bearer sk_test_demo_merchant"}


@pytest.fixture
def fraud_on(monkeypatch, service_clients, fraud_model_prefix):
    """Turn the flag on and route the gateway's fraud calls to a real, in-process fraud
    service (fakeredis for its windows)."""
    monkeypatch.setattr(settings, "fraud_check_enabled", True)
    monkeypatch.setattr(settings, "fraud_model_path", str(fraud_model_prefix))
    monkeypatch.setattr(fraud_main, "_redis", fakeredis.FakeRedis(decode_responses=True))
    with TestClient(fraud_main.app) as fraud_client:
        service_clients[settings.fraud_url] = fraud_client
        yield


@pytest.fixture
def acquirer_calls(monkeypatch):
    calls = []
    routed = gateway_routes.post_with_retry

    def counting(url, json, **kwargs):
        if url.startswith(f"{settings.acquirer_url}/v1/authorize"):
            calls.append(json["card_number"])
        return routed(url, json, **kwargs)

    monkeypatch.setattr(gateway_routes, "post_with_retry", counting)
    return calls


@pytest.fixture
def score_calls(monkeypatch):
    calls = []
    real = gateway_fraud.score

    def spy(req):
        calls.append(req)
        return real(req)

    monkeypatch.setattr(gateway_fraud, "score", spy)
    return calls


@pytest.fixture
def outcomes(monkeypatch):
    reported = []
    monkeypatch.setattr(
        gateway_fraud, "report_outcome", lambda req, declined: reported.append(declined)
    )
    return reported


def _stub_fraud_risk(risk, threshold=0.9):
    stub = FraudModel(model=None, meta={"model": "stub", "threshold": threshold, "features": []})
    stub.predict = lambda features, amount: risk
    fraud_main.app.state.model = stub


def _pay(client, card_number, amount_cents, headers=None, **customer):
    token = client.post("/v1/tokens", json={"card_number": card_number}).json()["token"]
    resp = client.post(
        "/v1/payment_intents",
        json={"amount_cents": amount_cents, "token": token, **customer},
        headers={**AUTH_HEADERS, **(headers or {})},
    )
    assert resp.status_code == 200
    return resp.json()


def _fraud_row(intent_id):
    with get_session() as session:
        return session.get(FraudCheck, intent_id)


def test_flag_off_never_calls_fraud_service(gateway_client, monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("fraud service called with the flag off")

    monkeypatch.setattr(gateway_fraud, "score", must_not_be_called)
    body = _pay(gateway_client, "4242424242424242", 1500)
    assert body["status"] == "succeeded"
    assert body["fraud_decision"] is None
    assert _fraud_row(body["id"]) is None


def test_normal_payment_is_scored_and_allowed(gateway_client, fraud_on):
    body = _pay(
        gateway_client, "4242424242424242", 4200,
        customer_ip="198.51.100.7", customer_email="shopper@example.com",
    )
    assert body["status"] == "succeeded"
    assert body["fraud_decision"] == "allow"
    row = _fraud_row(body["id"])
    assert row is not None and row.decision == "allow"


def test_card_testing_burst_is_blocked_before_the_acquirer(
    gateway_client, fraud_on, acquirer_calls
):
    rng = random.Random(0)
    results = [
        _pay(
            gateway_client,
            complete_luhn("424242" + f"{rng.randrange(10**9):09d}"),
            100,
            customer_ip="203.0.113.66",
            customer_email="tester@example.com",
        )
        for _ in range(15)
    ]
    blocked = [r for r in results if r["decline_reason"] == "fraud_suspected"]
    assert results[-1]["decline_reason"] == "fraud_suspected"
    assert len(blocked) >= 10
    assert all(r["status"] == "failed" and r["fraud_decision"] == "block" for r in blocked)
    # Blocked attempts never reach the acquirer.
    assert len(acquirer_calls) == len(results) - len(blocked)


def test_fraud_service_down_fails_open(gateway_client, fraud_on, monkeypatch):
    routed = gateway_fraud.post_with_retry

    def fraud_down(url, json, **kwargs):
        if url.startswith(settings.fraud_url):
            raise UpstreamError("fraud service timed out")
        return routed(url, json, **kwargs)

    monkeypatch.setattr(gateway_fraud, "post_with_retry", fraud_down)
    body = _pay(gateway_client, "4242424242424242", 1500, customer_ip="198.51.100.8")
    assert body["status"] == "succeeded"
    assert body["fraud_decision"] is None
    assert _fraud_row(body["id"]) is None


def test_idempotent_replay_is_scored_once(gateway_client, fraud_on, score_calls):
    key = {"Idempotency-Key": f"fraud-replay-{uuid.uuid4().hex[:8]}"}
    first = _pay(gateway_client, "4242424242424242", 1500, headers=key)
    second = _pay(gateway_client, "4242424242424242", 1500, headers=key)
    assert first["id"] == second["id"]
    assert second["fraud_decision"] == "allow"
    # A replay must not count as a second attempt in the fraud windows.
    assert len(score_calls) == 1


def test_outcome_reported_for_issuer_answers(gateway_client, fraud_on, outcomes):
    _pay(gateway_client, "4242424242424242", 1500, customer_ip="198.51.100.20")
    _pay(gateway_client, "4000000000009995", 500, customer_ip="198.51.100.21")
    assert outcomes == [False, True]


def test_blocked_payment_is_reported_as_declined(gateway_client, fraud_on, outcomes):
    _stub_fraud_risk(0.99)
    body = _pay(gateway_client, "4242424242424242", 1500)
    assert body["decline_reason"] == "fraud_suspected"
    # Otherwise an IP's decline rate would sink toward 0 the more it gets blocked.
    assert outcomes == [True]


def test_no_outcome_when_issuer_answer_is_lost(gateway_client, fraud_on, outcomes, monkeypatch):
    monkeypatch.setattr(settings, "simulate_lost_response_delay_seconds", 0.05)
    body = _pay(
        gateway_client, "4242424242424242", 1000, headers={"X-Simulate-Lost-Response": "true"}
    )
    assert body["decline_reason"] == "upstream_timeout"
    assert outcomes == []


def test_review_still_reaches_the_acquirer(gateway_client, fraud_on, acquirer_calls):
    _stub_fraud_risk(0.6)
    body = _pay(gateway_client, "4242424242424242", 1500)
    assert body["fraud_decision"] == "review"
    assert body["status"] == "succeeded"
    assert len(acquirer_calls) == 1


def test_missing_shopper_ip_gets_its_own_key(gateway_client, fraud_on, score_calls):
    # Never the merchant server's address: that would lump all its shoppers together.
    first = _pay(gateway_client, "4242424242424242", 1500)
    second = _pay(gateway_client, "4242424242424242", 1500)
    assert score_calls[0].ip == f"none:{first['id']}"
    assert score_calls[0].ip != score_calls[1].ip
    assert second["fraud_decision"] == "allow"


def test_card_too_short_for_a_bin_fails_open(gateway_client, fraud_on):
    # "18" passes the Luhn check but can't produce a 6-digit BIN for the fraud request.
    body = _pay(gateway_client, "18", 1500)
    assert body["fraud_decision"] is None
    assert body["decline_reason"] != "fraud_suspected"


def test_fraud_decision_is_returned_when_listing(gateway_client, fraud_on):
    body = _pay(gateway_client, "4242424242424242", 1500, customer_ip="198.51.100.30")
    listed = {i["id"]: i for i in gateway_client.get("/v1/payment_intents").json()}
    assert listed[body["id"]]["fraud_decision"] == "allow"


def test_malformed_fraud_response_fails_open(gateway_client, fraud_on, monkeypatch):
    routed = gateway_fraud.post_with_retry

    def garbage(url, json, **kwargs):
        if url.startswith(settings.fraud_url):
            return httpx.Response(200, text="<html>oops</html>")
        return routed(url, json, **kwargs)

    monkeypatch.setattr(gateway_fraud, "post_with_retry", garbage)
    body = _pay(gateway_client, "4242424242424242", 1500)
    assert body["status"] == "succeeded"
    assert body["fraud_decision"] is None
    assert _fraud_row(body["id"]) is None
