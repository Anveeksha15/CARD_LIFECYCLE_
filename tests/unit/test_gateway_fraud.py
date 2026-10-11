import httpx
import pytest
import respx

from common.config import settings
from services.gateway.fraud import build_request, fingerprint, report_outcome, score

SCORE_URL = f"{settings.fraud_url}/v1/score"


def test_fingerprint_is_stable_and_card_specific():
    assert fingerprint("4242424242424242") == fingerprint("4242424242424242")
    assert fingerprint("4242424242424242") != fingerprint("5555555555554444")
    assert "4242424242424242" not in fingerprint("4242424242424242")


def test_request_hides_secrets_and_isolates_missing_email():
    req = build_request("4242424242424242", 100, "1.2.3.4", None, "pi_1", "sk_test_secret")
    assert req.bin == "424242"
    assert "sk_test_secret" not in req.model_dump_json()
    assert "4242424242424242" not in req.model_dump_json()
    other = build_request("4242424242424242", 100, "1.2.3.4", None, "pi_2", "sk_test_secret")
    assert req.email != other.email


# Fail open: every failure mode returns None after exactly one attempt.


def _req():
    return build_request("4242424242424242", 4200, "1.2.3.4", "a@x", "pi_1", "sk_test")


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("down"),
        httpx.Response(500),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"risk": "high"}),
    ],
    ids=["timeout", "connect-error", "http-500", "non-json", "wrong-shape"],
)
@respx.mock
def test_score_fails_open_after_one_attempt(outcome):
    route = respx.post(SCORE_URL)
    if isinstance(outcome, Exception):
        route.side_effect = outcome
    else:
        route.return_value = outcome
    assert score(_req()) is None
    assert route.call_count == 1


@respx.mock
def test_score_uses_the_fraud_timeout_budget():
    route = respx.post(SCORE_URL).mock(
        return_value=httpx.Response(
            200, json={"risk": 0.1, "decision": "allow", "model": "gbm", "features": {}}
        )
    )
    assert score(_req()).decision == "allow"
    timeout = route.calls.last.request.extensions["timeout"]
    assert timeout["read"] == settings.fraud_timeout_seconds


@respx.mock
def test_report_outcome_swallows_errors():
    respx.post(f"{settings.fraud_url}/v1/outcome").side_effect = httpx.ConnectError("down")
    report_outcome(_req(), declined=True)  # must not raise from a background task
