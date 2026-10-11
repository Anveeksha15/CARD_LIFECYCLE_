# card-lifecycle-sim

A small backend system that reproduces the Stripe-like card payment lifecycle in code.
Each party — a gateway, an acquiring bank, a processor, a card network, and an issuing
bank — is its own FastAPI service. A payment travels through all of them over HTTP in
real time, then gets settled "days later" (compressed to seconds) by a background
worker that records the money movement in a double-entry ledger.

On top of that lifecycle sits a **real-time fraud check**: before a payment reaches the
acquirer, the gateway asks a scoring service (Redis sliding-window features + a
gradient-boosting model) whether it looks like card testing, and blocks the bursts.
See [Fraud detection](#fraud-detection).

**Rule:** only fake card numbers are used — Luhn-valid numbers generated for this
project, or Stripe's own [published test cards](https://stripe.com/docs/testing).
Never real card data.

## Quickstart

```bash
uv sync --all-groups
docker compose up --build -d
docker exec card_lifecycle-issuer-1 python -m scripts.seed_issuer
uv run python -m scripts.demo
```

That last command walks through everything the project promises: a payment traveling
through all five services, a decline, a retried request that doesn't double-charge,
settlement landing a few seconds later, and a card-testing burst getting blocked while a
normal customer still pays.

Run the test suite (no Docker required — see [docs/decisions.md](docs/decisions.md); the
two idempotency tests need Redis, e.g. `docker compose up -d redis`):

```bash
uv run ruff check .
uv run pytest -q
```

## Architecture

```mermaid
sequenceDiagram
    participant Merchant
    participant Gateway
    participant Fraud
    participant Acquirer
    participant Processor
    participant Network
    participant Issuer

    Merchant->>Gateway: POST /v1/tokens {card_number}
    Gateway-->>Merchant: {token}
    Merchant->>Gateway: POST /v1/payment_intents {amount, token}<br/>Idempotency-Key, Authorization
    Gateway->>Gateway: seen this Idempotency-Key? return cached result
    opt FRAUD_CHECK_ENABLED (150 ms budget, fails open)
        Gateway->>Fraud: POST /v1/score (card fingerprint, BIN, IP, email, amount)
        Fraud-->>Gateway: allow | review | block (block = declined, stop here)
    end
    Gateway->>Acquirer: POST /v1/authorize (card, amount, request_id)
    Acquirer->>Acquirer: is the merchant's api_key valid?
    Acquirer->>Processor: POST /v1/authorize
    Processor->>Processor: BIN prefix -> network
    Processor->>Network: POST /v1/authorize
    Network->>Issuer: POST /v1/authorize
    Issuer->>Issuer: card status + balance -> approve/decline
    Issuer-->>Network: approved | declined(reason)
    Network-->>Processor: ...
    Processor-->>Acquirer: ...
    Acquirer-->>Gateway: ...
    Gateway-->>Merchant: payment_intent {status, decline_reason}
    Gateway-)Merchant: signed webhook (background task)
```

```mermaid
sequenceDiagram
    participant Gateway
    participant Settlement
    participant Ledger as ledger.db (settlement)

    loop every settlement_poll_interval_seconds
        Settlement->>Gateway: GET /v1/payment_intents?unsettled=true
        Settlement->>Settlement: elapsed >= settlement_delay_seconds?
        Settlement->>Ledger: debit issuer_clearing / credit acquirer_clearing (full amount)
        Settlement->>Ledger: debit acquirer_clearing / credit merchant_payable (amount - fee)
        Settlement->>Ledger: debit acquirer_clearing / credit platform_fee_revenue (fee)
        Settlement->>Gateway: POST /v1/payment_intents/{id}/settle
    end
```

See [docs/architecture.md](docs/architecture.md) for the fuller walkthrough (including
the reversal path) and [docs/decisions.md](docs/decisions.md) for why things are built
the way they are.

## Fraud detection

The gateway scores each payment with `services/fraud` before authorizing it. The service
keeps sliding windows in Redis: attempts, declines and distinct cards per IP, plus
attempts per email and per BIN. It feeds them with the amount to a gradient-boosting
model and answers `allow`, `review` or `block`. The model was trained offline on
synthetic traffic with four kinds of card-testing attacker, and compared with a rules
baseline on a time-based test split:

| on synthetic data, test slice | F1 (seed 42) | F1 (5 more seeds) | false-positive rate (seed 42) |
|---|---|---|---|
| rules baseline | 0.54 | 0.53 ± 0.11 | 1.24% |
| gradient boosting | 0.95 | 0.87 ± 0.08 | 0.12% |

- The check adds **~8 ms p50** to a payment. `/v1/score` alone is about 5 ms p50 and 6 ms p95.
- It sits behind `FRAUD_CHECK_ENABLED` and **fails open** with a 150 ms budget, so a
  fraud-service outage never becomes a payment outage.
- The same features are computed offline for training and live in Redis for serving, and
  a parity test keeps the two identical.

**Limitations, briefly:** the data is synthetic and small, I wrote both the attackers
and the features, and results move noticeably between dataset seeds. On a flash sale it
never trained on, the model flagged 1.4% of buyers on one seed and 30.6% on another. The full
write-up, with both failure cases and every limitation, is
**[docs/fraud.md](docs/fraud.md)**.

## Project layout

```
common/        shared schemas, Luhn check, HMAC signing, retrying HTTP client, config
services/
  gateway/     tokens, payment_intents, idempotency, webhooks, its own SQLite db
  acquirer/    merchant validation
  processor/   BIN -> network routing
  network/     routes to the issuer
  issuer/      account status/balance checks, authorization + reversal records
  settlement/  background worker + double-entry ledger
  fraud/       /v1/score + /v1/outcome, Redis window features, rules baseline, model loader
ml/            synthetic data, offline features, training, evaluation, results/
merchant_demo/ a "merchant" exercising the API, and a webhook receiver
scripts/       seed_issuer.py, demo.py (the end-to-end walkthrough above),
               fraud_latency.py (p50/p95 of the fraud check)
tests/         unit tests + in-process integration tests (see docs/decisions.md)
```

## What's excluded (future work)

Per the original scope, these are deliberately left out:

- Encryption at rest (card/token data sits in plain SQLite columns)
- Authorization and capture as separate steps (this sim auto-captures immediately)
- Chargebacks / disputes
- Multi-currency (everything is an opaque integer `amount_cents`)
- A webhook retry queue (delivery is best-effort, single attempt)
- A flat per-transaction fee component (the settlement fee is percentage-only)
