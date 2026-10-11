"""End-to-end demo against an already-running stack (docker compose up --build -d).

Walks through everything the project promises: a payment traveling through all
five services, a decline, a retried request that doesn't double-charge, settlement
landing later, and a card-testing burst stopped by the fraud check.
"""

import random
import time
import uuid

import httpx

from common.config import settings
from common.luhn import complete_luhn
from merchant_demo.checkout import GATEWAY_URL, buy

# Documentation-only IP ranges (RFC 5737), so demo traffic never looks like a real address.
NORMAL_IP = "198.51.100.{}"
ATTACKER_IP = "203.0.113.66"


def shopper(n: int) -> dict:
    # Each demo customer gets their own IP and email, like real shoppers - otherwise every
    # demo payment shares one IP and repeated runs start to look like card testing.
    return {"customer_ip": NORMAL_IP.format(n), "customer_email": f"shopper{n}@example.com"}


def card_testing_burst(attempts: int = 15) -> None:
    rng = random.Random()
    blocked = 0
    for i in range(1, attempts + 1):
        card = complete_luhn("424242" + f"{rng.randrange(10**9):09d}")  # fake, never issued
        r = buy(card, 100, customer_ip=ATTACKER_IP, customer_email="tester@example.com",
                quiet=True)
        blocked += r["decline_reason"] == "fraud_suspected"
        print(f"  attempt {i:2d}: card ...{card[-4:]}  fraud={r['fraud_decision']!s:<6} "
              f"status={r['status']:<9} reason={r['decline_reason']}")
    print(f"blocked {blocked}/{attempts} attempts before they reached the acquirer")
    if blocked == 0:
        print("no blocks - is FRAUD_CHECK_ENABLED=true on the gateway and the fraud service up?")


def main() -> None:
    n = random.randrange(1, 200)

    print("=== 1. Happy path (Visa) ===")
    approved = buy("4242424242424242", 1500, **shopper(n))

    print("\n=== 2. Decline (insufficient funds) ===")
    buy("4000000000009995", 500, **shopper(n + 1))

    print("\n=== 3. Idempotent replay (same Idempotency-Key twice) ===")
    # Unique per run so repeat demos don't replay a cached result from Redis
    # left over from an earlier run - the within-run replay is what matters.
    idempotency_key = f"demo-idem-key-{uuid.uuid4().hex[:8]}"
    first = buy("5555555555554444", 2000, idempotency_key=idempotency_key, **shopper(n + 2))
    second = buy("5555555555554444", 2000, idempotency_key=idempotency_key, **shopper(n + 2))
    assert first["id"] == second["id"], "replay should return the same payment_intent"
    print(f"confirmed: both calls returned {first['id']} - no double charge")

    wait_seconds = settings.settlement_delay_seconds + 2
    print(f"\n=== 4. Waiting {wait_seconds}s for settlement to land ===")
    time.sleep(wait_seconds)

    unsettled_ids = {
        intent["id"]
        for intent in httpx.get(
            f"{GATEWAY_URL}/v1/payment_intents", params={"unsettled": "true"}
        ).json()
    }
    if approved["id"] in unsettled_ids:
        print(f"{approved['id']} is still unsettled - give the settlement worker more time")
    else:
        print(f"{approved['id']} has settled - check settlement.db inside the settlement "
              "container for its ledger entries")

    print("\n=== 5. Card-testing burst: one IP, a new card every attempt, $1 each ===")
    card_testing_burst()

    print("\n=== 6. Meanwhile, a normal customer still gets through ===")
    r = buy("4242424242424242", 4200, **shopper(n + 3), quiet=True)
    print(f"  status={r['status']}  fraud={r['fraud_decision']}")


if __name__ == "__main__":
    main()
