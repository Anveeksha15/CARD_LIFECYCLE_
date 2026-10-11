"""Acts as the merchant: tokenizes a card, then creates a payment_intent."""

import argparse

import httpx

GATEWAY_URL = "http://localhost:8000"
API_KEY = "sk_test_demo_merchant"


def buy(
    card_number: str,
    amount_cents: int,
    *,
    idempotency_key: str | None = None,
    simulate_lost_response: bool = False,
    customer_ip: str | None = None,
    customer_email: str | None = None,
    quiet: bool = False,
) -> dict:
    token_resp = httpx.post(f"{GATEWAY_URL}/v1/tokens", json={"card_number": card_number})
    token_resp.raise_for_status()
    token = token_resp.json()["token"]
    if not quiet:
        print(f"tokenized card -> {token}")

    headers = {"Authorization": f"Bearer {API_KEY}"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    if simulate_lost_response:
        headers["X-Simulate-Lost-Response"] = "true"

    intent_resp = httpx.post(
        f"{GATEWAY_URL}/v1/payment_intents",
        json={
            "amount_cents": amount_cents,
            "token": token,
            "customer_ip": customer_ip,
            "customer_email": customer_email,
        },
        headers=headers,
        timeout=10.0,
    )
    intent_resp.raise_for_status()
    result = intent_resp.json()
    if not quiet:
        print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card_number", nargs="?", default="4242424242424242")
    parser.add_argument("amount_cents", nargs="?", type=int, default=1500)
    parser.add_argument("--idempotency-key")
    parser.add_argument("--simulate-lost-response", action="store_true")
    args = parser.parse_args()

    buy(
        args.card_number,
        args.amount_cents,
        idempotency_key=args.idempotency_key,
        simulate_lost_response=args.simulate_lost_response,
    )
