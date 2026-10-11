"""Measure fraud-check latency against the running stack (docker compose up --build -d).

Usage: python -m scripts.fraud_latency [--score-n 500] [--gateway-n 200]

Two numbers: the /v1/score call on its own, and a whole gateway payment intent (which
includes the fraud check when FRAUD_CHECK_ENABLED=true). Run it once with the flag on
and once with `FRAUD_CHECK_ENABLED=false docker compose up -d gateway` to see the cost.
"""

import argparse
import statistics
import time
import uuid
from pathlib import Path

import httpx

from merchant_demo.checkout import API_KEY

# 127.0.0.1, not "localhost": on Windows + Docker Desktop, POSTs to "localhost" pick up
# ~45 ms of host-side overhead (even for a no-op endpoint) that the services never see -
# the gateway reaches the fraud service over the Docker network.
FRAUD_URL = "http://127.0.0.1:8005"
GATEWAY_URL = "http://127.0.0.1:8000"
RESULTS = Path("ml/results/latency.md")


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    q = statistics.quantiles(samples_ms, n=100)
    return {"p50": q[49], "p95": q[94], "p99": q[98], "max": max(samples_ms)}


def time_score(client: httpx.Client, n: int) -> list[float]:
    run = uuid.uuid4().hex[:6]
    samples = []
    for i in range(n):
        body = {
            "event_id": f"lat_{run}_{i}",
            "card_fingerprint": f"fp_{run}_{i}",
            "bin": "424242",
            "ip": f"198.51.100.{i % 250}",
            "email": f"lat{run}{i}@example.com",
            "amount_cents": 4200,
            "merchant_id": "m_latency",
        }
        start = time.perf_counter()
        client.post(f"{FRAUD_URL}/v1/score", json=body).raise_for_status()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def time_gateway(client: httpx.Client, n: int) -> tuple[list[float], bool]:
    samples, scored = [], False
    for i in range(n):
        token = client.post(
            f"{GATEWAY_URL}/v1/tokens", json={"card_number": "4242424242424242"}
        ).json()["token"]
        body = {
            "amount_cents": 1000,
            "token": token,
            "customer_ip": f"198.51.100.{i % 250}",
            "customer_email": f"gw{uuid.uuid4().hex[:8]}@example.com",
        }
        start = time.perf_counter()
        resp = client.post(
            f"{GATEWAY_URL}/v1/payment_intents",
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        samples.append((time.perf_counter() - start) * 1000)
        resp.raise_for_status()
        scored = scored or resp.json()["fraud_decision"] is not None
    return samples, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score-n", type=int, default=500)
    parser.add_argument("--gateway-n", type=int, default=200)
    args = parser.parse_args()

    with httpx.Client(timeout=10.0) as client:
        for _ in range(20):  # warm up connections and the model
            client.get(f"{FRAUD_URL}/health")
        score = percentiles(time_score(client, args.score_n))
        gateway_samples, scored = time_gateway(client, args.gateway_n)
        gateway = percentiles(gateway_samples)

    label = "on" if scored else "off"
    lines = [
        "| measurement | n | p50 ms | p95 ms | p99 ms | max ms |",
        "|---|---|---|---|---|---|",
        f"| /v1/score (direct) | {args.score_n} | {score['p50']:.1f} | {score['p95']:.1f} "
        f"| {score['p99']:.1f} | {score['max']:.1f} |",
        f"| gateway payment intent, fraud check {label} | {args.gateway_n} "
        f"| {gateway['p50']:.1f} | {gateway['p95']:.1f} | {gateway['p99']:.1f} "
        f"| {gateway['max']:.1f} |",
    ]
    table = "\n".join(lines)
    print(table)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(f"\n_run {time.strftime('%Y-%m-%d %H:%M')}, fraud check {label}_\n\n")
        f.write(table + "\n")
    print(f"appended to {RESULTS}")


if __name__ == "__main__":
    main()
