import httpx
import pytest
import respx

import common.http as http
from common.http import UpstreamError, post_with_retry

URL = "http://upstream.test/v1/authorize"


@respx.mock
def test_succeeds_after_transient_failures():
    route = respx.post(URL)
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(200, json={"approved": True}),
    ]

    resp = post_with_retry(URL, {"amount_cents": 100}, max_attempts=3, backoff_base=0.01)

    assert resp.status_code == 200
    assert route.call_count == 3


@respx.mock
def test_raises_after_exhausting_attempts():
    respx.post(URL).mock(return_value=httpx.Response(503))

    with pytest.raises(UpstreamError):
        post_with_retry(URL, {"amount_cents": 100}, max_attempts=2, backoff_base=0.01)


@respx.mock
def test_calls_reuse_one_pooled_client(monkeypatch):
    calls = []
    real_post = http._client.post

    def counting_post(*args, **kwargs):
        calls.append(1)
        return real_post(*args, **kwargs)

    monkeypatch.setattr(http._client, "post", counting_post)
    respx.post(URL).mock(return_value=httpx.Response(200))

    post_with_retry(URL, {}, max_attempts=1)
    post_with_retry(URL, {}, max_attempts=1)
    assert len(calls) == 2
