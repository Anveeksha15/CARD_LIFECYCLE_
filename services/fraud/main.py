import logging
import time
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException, Request

from common.config import settings
from common.models import (
    FraudDecision,
    FraudOutcomeRequest,
    FraudScoreRequest,
    FraudScoreResponse,
)
from services.fraud.features import Tx, observe, record_outcome
from services.fraud.model import FraudModel

_redis = redis.from_url(settings.redis_url, decode_responses=True)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = FraudModel.load(settings.fraud_model_path)
    if settings.fraud_review_threshold >= app.state.model.threshold:
        # The block threshold comes from training, the review one from config - if they
        # cross, "review" can never be returned.
        log.warning(
            "fraud_review_threshold %.3f >= model block threshold %.3f: no review band",
            settings.fraud_review_threshold,
            app.state.model.threshold,
        )
    yield


app = FastAPI(title="fraud", lifespan=lifespan)


def _redis_unavailable(exc: redis.RedisError) -> HTTPException:
    # A clear 503 rather than a 500 traceback; the gateway fails open on either.
    return HTTPException(status_code=503, detail=f"feature store unavailable: {exc}")


def decide(risk: float, block_threshold: float) -> FraudDecision:
    if risk >= block_threshold:
        return FraudDecision.BLOCK
    if risk >= settings.fraud_review_threshold:
        return FraudDecision.REVIEW
    return FraudDecision.ALLOW


@app.get("/health")
def health(request: Request) -> dict:
    model: FraudModel = request.app.state.model
    return {"status": "ok", "service": "fraud", "model": model.name, "threshold": model.threshold}


@app.post("/v1/score", response_model=FraudScoreResponse)
def score(req: FraudScoreRequest, request: Request) -> FraudScoreResponse:
    model: FraudModel = request.app.state.model
    tx = Tx(req.event_id, req.card_fingerprint, req.bin, req.ip, req.email)
    # Records this attempt before scoring it, so a blocked attempt still counts.
    try:
        features = observe(_redis, tx, now=time.time())
    except redis.RedisError as exc:
        raise _redis_unavailable(exc) from exc
    risk = model.predict(features, req.amount_cents)
    return FraudScoreResponse(
        risk=risk, decision=decide(risk, model.threshold), model=model.name, features=features
    )


@app.post("/v1/outcome")
def outcome(req: FraudOutcomeRequest) -> dict:
    tx = Tx(req.event_id, req.card_fingerprint, req.bin, req.ip, req.email)
    try:
        record_outcome(_redis, tx, declined=req.declined)
    except redis.RedisError as exc:
        raise _redis_unavailable(exc) from exc
    return {"recorded": True}
