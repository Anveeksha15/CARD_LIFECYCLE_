import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from sqlmodel import select

from common.config import settings
from common.http import UpstreamError, post_with_retry
from common.models import (
    AcquirerAuthorizeRequest,
    AuthorizeResponse,
    DeclineReason,
    FraudDecision,
    PaymentIntentCreateRequest,
    PaymentIntentResponse,
    ReversalRequest,
    TokenizeRequest,
    TokenizeResponse,
)
from services.gateway import fraud
from services.gateway.db import FraudCheck, PaymentIntent, get_session
from services.gateway.idempotency import IdempotencyInProgress, claim, store_result
from services.gateway.vault import InvalidCardError, detokenize, tokenize
from services.gateway.webhooks import send_webhook

router = APIRouter()


def _to_response(intent: PaymentIntent, fraud_decision: str | None = None) -> PaymentIntentResponse:
    return PaymentIntentResponse(
        id=intent.id,
        amount_cents=intent.amount_cents,
        status=intent.status,
        decline_reason=intent.decline_reason,
        settled=intent.settled,
        created_at=intent.created_at,
        fraud_decision=fraud_decision,
    )


def _fraud_decisions(session, intent_ids: list[str]) -> dict[str, str]:
    if not intent_ids:
        return {}
    rows = session.exec(select(FraudCheck).where(FraudCheck.intent_id.in_(intent_ids))).all()
    return {row.intent_id: row.decision for row in rows}


@router.post("/v1/tokens", response_model=TokenizeResponse)
def create_token(req: TokenizeRequest) -> TokenizeResponse:
    try:
        token = tokenize(req.card_number)
    except InvalidCardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TokenizeResponse(token=token)


@router.get("/v1/payment_intents", response_model=list[PaymentIntentResponse])
def list_payment_intents(unsettled: bool = False) -> list[PaymentIntentResponse]:
    with get_session() as session:
        intents = session.exec(
            select(PaymentIntent).where(PaymentIntent.status == "succeeded")
        ).all()
        if unsettled:
            intents = [i for i in intents if not i.settled]
        decisions = _fraud_decisions(session, [i.id for i in intents])
    return [_to_response(i, decisions.get(i.id)) for i in intents]


@router.post("/v1/payment_intents/{intent_id}/settle", response_model=PaymentIntentResponse)
def settle_payment_intent(intent_id: str) -> PaymentIntentResponse:
    with get_session() as session:
        intent = session.get(PaymentIntent, intent_id)
        if intent is None:
            raise HTTPException(status_code=404, detail="payment_intent not found")
        intent.settled = True
        session.add(intent)
        session.commit()
        session.refresh(intent)
        return _to_response(intent, _fraud_decisions(session, [intent_id]).get(intent_id))


@router.post("/v1/payment_intents", response_model=PaymentIntentResponse)
def create_payment_intent(
    req: PaymentIntentCreateRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(..., alias="Authorization"),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    simulate_lost_response: bool = Header(False, alias="X-Simulate-Lost-Response"),
) -> PaymentIntentResponse:
    api_key = authorization.removeprefix("Bearer ").strip()

    idem_redis_key = f"idempotency:{api_key}:{idempotency_key}" if idempotency_key else None
    if idem_redis_key:
        try:
            cached = claim(idem_redis_key)
        except IdempotencyInProgress as exc:
            raise HTTPException(
                status_code=409,
                detail="a request with this idempotency key is already being processed",
            ) from exc
        if cached is not None:
            return PaymentIntentResponse(**cached)

    card_number = detokenize(req.token)
    if card_number is None:
        raise HTTPException(status_code=400, detail="unknown or expired token")

    intent_id = f"pi_{uuid.uuid4().hex[:16]}"
    request_id = uuid.uuid4().hex
    created_at = datetime.now(UTC)

    fraud_req = verdict = None
    if settings.fraud_check_enabled:
        fraud_req, verdict = fraud.check(  # verdict None = failed open
            card_number, req.amount_cents, req.customer_ip, req.customer_email, intent_id,
            api_key,
        )

    if verdict is not None and verdict.decision == FraudDecision.BLOCK:
        result = AuthorizeResponse(approved=False, reason=DeclineReason.FRAUD_SUSPECTED)
        # To the requester a block is a decline. Recording it as one keeps the per-IP
        # decline features close to the training data, where attacks are mostly declined;
        # otherwise an IP's decline rate would sink toward 0 the more it gets blocked.
        background_tasks.add_task(fraud.report_outcome, fraud_req, declined=True)
    else:
        payload = AcquirerAuthorizeRequest(
            api_key=api_key,
            card_number=card_number,
            amount_cents=req.amount_cents,
            request_id=request_id,
            simulate_lost_response=simulate_lost_response,
        )
        result = _authorize(payload)
        # An upstream timeout means the issuer's answer is unknown - don't record a guess.
        if verdict is not None and result.reason != DeclineReason.UPSTREAM_TIMEOUT:
            background_tasks.add_task(
                fraud.report_outcome, fraud_req, declined=not result.approved
            )

    status = "succeeded" if result.approved else "failed"

    with get_session() as session:
        session.add(
            PaymentIntent(
                id=intent_id,
                amount_cents=req.amount_cents,
                token=req.token,
                status=status,
                decline_reason=result.reason,
                request_id=request_id,
                created_at=created_at,
            )
        )
        if verdict is not None:
            session.add(
                FraudCheck(
                    intent_id=intent_id,
                    risk=verdict.risk,
                    decision=verdict.decision,
                    model=verdict.model,
                )
            )
        session.commit()

    response = PaymentIntentResponse(
        id=intent_id,
        amount_cents=req.amount_cents,
        status=status,
        decline_reason=result.reason,
        created_at=created_at,
        fraud_decision=verdict.decision if verdict is not None else None,
    )

    if idem_redis_key:
        store_result(idem_redis_key, response.model_dump(mode="json"))

    background_tasks.add_task(
        send_webhook,
        settings.merchant_webhook_url,
        {
            "type": "payment_intent.succeeded" if result.approved else "payment_intent.failed",
            "id": intent_id,
            "amount_cents": req.amount_cents,
            "status": status,
            "decline_reason": result.reason,
        },
    )

    return response


def _authorize(payload: AcquirerAuthorizeRequest) -> AuthorizeResponse:
    try:
        resp = post_with_retry(f"{settings.acquirer_url}/v1/authorize", payload.model_dump())
        return AuthorizeResponse.model_validate(resp.json())
    except UpstreamError:
        # We don't know whether the issuer approved before the response was
        # lost - best-effort reverse whatever it may have recorded for this
        # request_id, then surface it to the merchant as a plain failure.
        try:
            post_with_retry(
                f"{settings.acquirer_url}/v1/reversals",
                ReversalRequest(request_id=payload.request_id).model_dump(),
                max_attempts=1,
            )
        except UpstreamError:
            pass
        return AuthorizeResponse(approved=False, reason=DeclineReason.UPSTREAM_TIMEOUT)
