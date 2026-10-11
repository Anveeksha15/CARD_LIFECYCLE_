import time

import httpx

from common.config import settings

# One pooled client per process: reusing keep-alive connections skips a TCP handshake on
# every inter-service hop. httpx.Client is safe to share across FastAPI's worker threads.
_client = httpx.Client()


class UpstreamError(Exception):
    pass


def post_with_retry(
    url: str,
    json: dict,
    *,
    timeout: float | None = None,
    max_attempts: int | None = None,
    backoff_base: float | None = None,
) -> httpx.Response:
    timeout = settings.request_timeout_seconds if timeout is None else timeout
    max_attempts = settings.max_retry_attempts if max_attempts is None else max_attempts
    backoff_base = settings.retry_backoff_base_seconds if backoff_base is None else backoff_base

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _client.post(url, json=json, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(backoff_base * (2 ** (attempt - 1)))

    raise UpstreamError(f"failed to reach {url} after {max_attempts} attempts") from last_exc
