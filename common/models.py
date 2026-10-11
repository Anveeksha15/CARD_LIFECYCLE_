from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Network(StrEnum):
    VISA = "visa"
    MASTERCARD = "mastercard"
    AMEX = "amex"


class DeclineReason(StrEnum):
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_BLOCKED = "card_blocked"
    CARD_NOT_FOUND = "card_not_found"
    UNSUPPORTED_NETWORK = "unsupported_network"
    INVALID_MERCHANT = "invalid_merchant"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    FRAUD_SUSPECTED = "fraud_suspected"


class FraudDecision(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class AuthorizeRequest(BaseModel):
    card_number: str
    amount_cents: int = Field(gt=0)
    request_id: str
    # Demo-only chaos flag: tells the issuer to approve, persist the
    # authorization, then sleep past the caller's timeout so the response
    # never arrives - simulating the "approved but response lost" failure.
    simulate_lost_response: bool = False


class NetworkAuthorizeRequest(AuthorizeRequest):
    network: Network


class AcquirerAuthorizeRequest(AuthorizeRequest):
    api_key: str


class AuthorizeResponse(BaseModel):
    approved: bool
    reason: DeclineReason | None = None
    network: Network | None = None


class TokenizeRequest(BaseModel):
    card_number: str


class TokenizeResponse(BaseModel):
    token: str


class PaymentIntentCreateRequest(BaseModel):
    amount_cents: int = Field(gt=0)
    token: str
    # The shopper's details, passed by the merchant for fraud scoring (like Stripe Radar).
    customer_ip: str | None = None
    customer_email: str | None = None


class PaymentIntentResponse(BaseModel):
    id: str
    amount_cents: int
    status: str
    decline_reason: DeclineReason | None = None
    settled: bool = False
    created_at: datetime
    # None when the fraud check is disabled or failed open.
    fraud_decision: FraudDecision | None = None


class ReversalRequest(BaseModel):
    request_id: str


class ReversalResponse(BaseModel):
    reversed: bool


class FraudScoreRequest(BaseModel):
    event_id: str
    # Must be stable per card (e.g. an HMAC of the card number) - a fresh vault token per
    # attempt would make every retry of the same card look like a new card.
    card_fingerprint: str
    bin: str = Field(min_length=6, max_length=6)
    ip: str
    email: str
    amount_cents: int = Field(gt=0)
    merchant_id: str


class FraudScoreResponse(BaseModel):
    risk: float
    decision: FraudDecision
    model: str
    features: dict[str, float]


class FraudOutcomeRequest(BaseModel):
    event_id: str
    card_fingerprint: str
    bin: str
    ip: str
    email: str
    declined: bool
