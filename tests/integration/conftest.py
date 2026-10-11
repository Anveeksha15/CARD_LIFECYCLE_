"""Runs the real service chain in-process by routing common.http.post_with_retry's
call sites to each service's own FastAPI TestClient instead of real network calls.

The "issuer approves but the response is lost" scenario is simulated
deterministically rather than via real timeouts: the network -> issuer hop makes
the real call (so the issuer genuinely persists the approval), then treats it as
lost whenever the request carries simulate_lost_response. This avoids needing
real wall-clock timeouts or background threads in the test suite, which earlier
caused flaky, timing-dependent failures (nested hops sharing one thread pool
could starve each other and spuriously time out ordinary, non-chaos requests).
"""

import pytest
from fastapi.testclient import TestClient

import services.acquirer.main as acquirer_main
import services.gateway.fraud as gateway_fraud
import services.gateway.routes as gateway_routes
import services.issuer.main as issuer_main
import services.network.main as network_main
import services.processor.main as processor_main
from common.config import settings
from common.http import UpstreamError
from services.gateway.main import app as gateway_app


@pytest.fixture
def gateway_client():
    return TestClient(gateway_app)


@pytest.fixture
def service_clients():
    return {
        settings.acquirer_url: TestClient(acquirer_main.app),
        settings.processor_url: TestClient(processor_main.app),
        settings.network_url: TestClient(network_main.app),
        settings.issuer_url: TestClient(issuer_main.app),
    }


@pytest.fixture(autouse=True)
def route_post_with_retry(monkeypatch, service_clients):
    def fake_post_with_retry(url, json, *, timeout=None, max_attempts=None, backoff_base=None):
        max_attempts = settings.max_retry_attempts if max_attempts is None else max_attempts

        base_url, client = next(
            (b, c) for b, c in service_clients.items() if url.startswith(b)
        )
        path = url[len(base_url) :]

        if base_url == settings.issuer_url and json.get("simulate_lost_response"):
            # Make the real call once - the issuer genuinely persists the
            # approval - then treat the response as lost. Deterministic, no
            # real network timeout needed to reproduce the scenario.
            client.post(path, json=json)
            raise UpstreamError(f"simulated lost response from {url}")

        last_exc: Exception | None = None
        for _ in range(max_attempts):
            resp = client.post(path, json=json)
            if resp.status_code < 500:
                return resp
            last_exc = Exception(f"status {resp.status_code}")

        raise UpstreamError(f"failed to reach {url} after {max_attempts} attempts") from last_exc

    monkeypatch.setattr(acquirer_main, "post_with_retry", fake_post_with_retry)
    monkeypatch.setattr(processor_main, "post_with_retry", fake_post_with_retry)
    monkeypatch.setattr(network_main, "post_with_retry", fake_post_with_retry)
    monkeypatch.setattr(gateway_routes, "post_with_retry", fake_post_with_retry)
    monkeypatch.setattr(gateway_fraud, "post_with_retry", fake_post_with_retry)
