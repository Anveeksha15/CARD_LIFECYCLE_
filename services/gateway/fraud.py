"""The gateway's side of the fraud check.

Fails open: if the fraud service is slow, down or returns garbage, score() returns None
and the payment carries on to the acquirer - a fraud-service outage must not become a
payment outage. See docs/decisions.md.
"""

import hashlib
import hmac
import logging

import httpx
from pydantic import ValidationError

from common.config import settings
from common.http import UpstreamError, post_with_retry
from common.models import FraudOutcomeRequest, FraudScoreRequest, FraudScoreResponse

log = logging.getLogger(__name__)

_FAIL_OPEN_ERRORS = (UpstreamError, httpx.HTTPError, ValueError, ValidationError)


def fingerprint(card_number: str) -> str:
    """Stable per card, unlike vault tokens (a new random one on every tokenize call)."""
    key = settings.card_fingerprint_secret.encode()
    return hmac.new(key, card_number.encode(), hashlib.sha256).hexdigest()[:32]


def build_request(
    card_number: str,
    amount_cents: int,
    ip: str | None,
    email: str | None,
    intent_id: str,
    api_key: str,
) -> FraudScoreRequest:
    return FraudScoreRequest(
        event_id=intent_id,
        card_fingerprint=fingerprint(card_number),
        bin=card_number[:6],
        # Missing shopper details get a key of their own. A shared placeholder - or the
        # merchant server's address, which is what request.client.host would be - would
        # make all of a merchant's shoppers look like one actor's burst.
        ip=ip or f"none:{intent_id}",
        email=email or f"none:{intent_id}",
        amount_cents=amount_cents,
        # The fraud service never sees the merchant's secret key.
        merchant_id="m_" + hashlib.sha256(api_key.encode()).hexdigest()[:12],
    )


def check(
    card_number: str,
    amount_cents: int,
    ip: str | None,
    email: str | None,
    intent_id: str,
    api_key: str,
) -> tuple[FraudScoreRequest | None, FraudScoreResponse | None]:
    """Build the request and score it. Anything that goes wrong fails open as (req, None)."""
    try:
        req = build_request(card_number, amount_cents, ip, email, intent_id, api_key)
    except ValueError as exc:  # e.g. a card too short to have a 6-digit BIN
        log.warning("fraud check failed open for %s: %s", intent_id, exc)
        return None, None
    return req, score(req)


def score(req: FraudScoreRequest) -> FraudScoreResponse | None:
    """Score a payment within the time budget, or return None (fail open)."""
    try:
        resp = post_with_retry(
            f"{settings.fraud_url}/v1/score",
            req.model_dump(),
            # One attempt only: a retry can't fit inside a 150 ms budget.
            timeout=settings.fraud_timeout_seconds,
            max_attempts=1,
        )
        return FraudScoreResponse.model_validate(resp.json())
    except _FAIL_OPEN_ERRORS as exc:
        log.warning("fraud check failed open for %s: %s", req.event_id, exc)
        return None


def report_outcome(req: FraudScoreRequest, declined: bool) -> None:
    """Tell the fraud service how the payment ended. Runs as a background task, so it gets
    the normal timeout and retries rather than the scoring budget."""
    body = FraudOutcomeRequest(
        event_id=req.event_id,
        card_fingerprint=req.card_fingerprint,
        bin=req.bin,
        ip=req.ip,
        email=req.email,
        declined=declined,
    )
    try:
        post_with_retry(
            f"{settings.fraud_url}/v1/outcome",
            body.model_dump(),
        )
    except _FAIL_OPEN_ERRORS as exc:
        log.warning("fraud outcome not recorded for %s: %s", req.event_id, exc)
