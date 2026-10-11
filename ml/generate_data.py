"""Synthetic payment traffic with labeled card-testing attacks.

Usage: python -m ml.generate_data [--n 100000] [--seed 42] [--out ml/data/transactions.csv]

Every row is one authorization attempt. Card numbers are never generated - only opaque
`card_token`s and a 6-digit `bin` drawn from Stripe test-card prefixes and a few made-up
ones, so no real card data exists anywhere in the dataset.

`scenario` and `campaign_id` describe how a row was generated. They exist for error
analysis only and must never be used as model features.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = 7
HORIZON_SECONDS = DAYS * 86_400

COLUMNS = [
    "event_id",
    "ts",
    "card_token",
    "bin",
    "amount_cents",
    "ip",
    "email",
    "merchant_id",
    "outcome",
    "is_attack",
    "scenario",
    "campaign_id",
]

# First three are prefixes of the Stripe test cards in common/luhn.py; the rest are made up.
BINS = ["424242", "555555", "378282", "400000", "411111", "510510", "601100", "353011"]
BIN_WEIGHTS = np.array([0.30, 0.22, 0.08, 0.12, 0.12, 0.08, 0.05, 0.03])

N_MERCHANTS = 50
FLASH_SALE_MERCHANT = "merchant_07"

# Share of events by source. Attacks land at roughly 3.5% of the total.
ATTACK_SHARE = 0.035
FLASH_SALE_SHARE = 0.012
NORMAL_DECLINE_RATE = 0.05
RETRY_AFTER_DECLINE_PROB = 0.4
# Amounts overlap on purpose so amount alone can't separate the classes: some customers
# make tiny purchases, and some attackers test cards with normal-looking amounts.
SMALL_PURCHASE_PROB = 0.03
ATTACK_MIMIC_PROB = 0.2

# Rough hour-of-day traffic shape for normal customers (index = hour).
HOURLY_WEIGHTS = np.array(
    [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1, 2, 3, 3.5, 4, 4, 4.5, 4, 4, 4, 4, 4.5, 5, 5.5, 5, 4, 3, 2]
)


class _Ids:
    """Random-looking identifiers drawn from one shared space, so an IP or email never
    reveals whether it belongs to a customer or an attacker."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def ip(self) -> str:
        a, b, c, d = self.rng.integers(1, 255, size=4)
        return f"{a}.{b}.{c}.{d}"

    def email(self) -> str:
        return f"u{self.rng.integers(0, 16**10):010x}@example.com"

    def card(self) -> str:
        return f"card_{self.rng.integers(0, 16**12):012x}"


def _row(ts, card, bin_, amount, ip, email, merchant, declined, is_attack, scenario, campaign):
    return {
        "ts": round(float(ts), 3),
        "card_token": card,
        "bin": bin_,
        "amount_cents": int(amount),
        "ip": ip,
        "email": email,
        "merchant_id": merchant,
        "outcome": "declined" if declined else "approved",
        "is_attack": int(is_attack),
        "scenario": scenario,
        "campaign_id": campaign,
    }


def _merchant(rng: np.random.Generator) -> str:
    # Zipf-ish popularity: a few big merchants, a long tail of small ones.
    weights = 1.0 / np.arange(1, N_MERCHANTS + 1)
    idx = rng.choice(N_MERCHANTS, p=weights / weights.sum())
    return f"merchant_{idx + 1:02d}"


def _normal_amount(rng: np.random.Generator) -> int:
    if rng.random() < SMALL_PURCHASE_PROB:
        return int(rng.integers(50, 501))  # tips, digital goods: $0.50 - $5.00
    # Lognormal around $40, clipped to $5-$300.
    return int(np.clip(rng.lognormal(mean=np.log(4000), sigma=0.8), 500, 30_000))


def _normal_ts(rng: np.random.Generator) -> float:
    hour_p = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()
    day = rng.integers(0, DAYS)
    hour = rng.choice(24, p=hour_p)
    return day * 86_400 + hour * 3_600 + rng.uniform(0, 3_600)


def _make_customers(rng: np.random.Generator, ids: _Ids, n_customers: int) -> list[dict]:
    # A few carrier-grade NAT / office IPs shared by many unrelated customers.
    shared_ips = [ids.ip() for _ in range(5)]
    customers = []
    for _ in range(n_customers):
        shared = rng.random() < 0.04
        n_cards = 1 if rng.random() < 0.75 else 2
        cards = [
            (ids.card(), BINS[rng.choice(len(BINS), p=BIN_WEIGHTS / BIN_WEIGHTS.sum())])
            for _ in range(n_cards)
        ]
        customers.append(
            {
                "ip": shared_ips[rng.integers(0, len(shared_ips))] if shared else ids.ip(),
                "email": ids.email(),
                "cards": cards,
                "scenario": "normal_shared_ip" if shared else "normal",
            }
        )
    return customers


def _normal_purchase(rng, customer, ts, merchant, scenario) -> list[dict]:
    """One purchase, plus the customer's retries if it gets declined."""
    card, bin_ = customer["cards"][rng.integers(0, len(customer["cards"]))]
    amount = _normal_amount(rng)
    declined = rng.random() < NORMAL_DECLINE_RATE
    rows = [_row(ts, card, bin_, amount, customer["ip"], customer["email"], merchant,
                 declined, 0, scenario, "")]
    if declined and rng.random() < RETRY_AFTER_DECLINE_PROB:
        # Retrying the same card a few times within about a minute - looks a bit like
        # card testing, but it's one card and normal-sized amounts.
        for _ in range(rng.integers(1, 4)):
            ts += rng.uniform(5, 25)
            declined = rng.random() < 0.6
            rows.append(_row(ts, card, bin_, amount, customer["ip"], customer["email"],
                             merchant, declined, 0, "normal_retry", ""))
            if not declined:
                break
    return rows


def _flash_sale(rng, ids: _Ids, customers: list[dict], n_events: int) -> list[dict]:
    """A 10-minute burst at one merchant. Many buyers are on the same mobile-carrier NAT IP,
    so per-IP counters spike exactly like an attack."""
    start = rng.uniform(0.75, 0.95) * HORIZON_SECONDS  # lands in the test period
    carrier_ip = ids.ip()
    rows = []
    for _ in range(n_events):
        customer = dict(customers[rng.integers(0, len(customers))])
        if rng.random() < 0.3:
            customer["ip"] = carrier_ip
        ts = start + rng.uniform(0, 600)
        rows.extend(_normal_purchase(rng, customer, ts, FLASH_SALE_MERCHANT, "flash_sale"))
    return rows


def _attack_amount(rng: np.random.Generator) -> int:
    if rng.random() < ATTACK_MIMIC_PROB:
        return _normal_amount(rng)
    return int(rng.integers(50, 201))  # $0.50 - $2.00


def _attack_burst(rng, ids: _Ids, campaign: str) -> list[dict]:
    """One IP and email firing 20-120 fresh cards in 1-5 minutes."""
    n = int(rng.integers(20, 121))
    start = rng.uniform(0, HORIZON_SECONDS - 600)
    times = np.sort(start + rng.uniform(0, rng.uniform(60, 300), size=n))
    ip, email, merchant = ids.ip(), ids.email(), _merchant(rng)
    fixed_bin = BINS[rng.integers(0, len(BINS))] if rng.random() < 0.7 else None
    return [
        _row(t, ids.card(), fixed_bin or BINS[rng.integers(0, len(BINS))],
             _attack_amount(rng), ip, email, merchant, rng.random() < 0.85, 1,
             "attack_burst", campaign)
        for t in times
    ]


def _attack_slow(rng, ids: _Ids, campaign: str) -> list[dict]:
    """Same pattern, but spread over 1-2 hours to stay under per-minute thresholds."""
    n = int(rng.integers(20, 81))
    start = rng.uniform(0, HORIZON_SECONDS - 8_000)
    times = np.sort(start + rng.uniform(0, rng.uniform(3_600, 7_200), size=n))
    ip, email, merchant = ids.ip(), ids.email(), _merchant(rng)
    bin_ = BINS[rng.integers(0, len(BINS))]
    return [
        _row(t, ids.card(), bin_, _attack_amount(rng), ip, email, merchant,
             rng.random() < 0.85, 1, "attack_slow", campaign)
        for t in times
    ]


def _attack_rotating_ip(rng, ids: _Ids, campaign: str) -> list[dict]:
    """A shared email and BIN, but every attempt comes from a proxy pool of IPs."""
    n = int(rng.integers(30, 101))
    start = rng.uniform(0, HORIZON_SECONDS - 900)
    times = np.sort(start + rng.uniform(0, rng.uniform(120, 600), size=n))
    pool = [ids.ip() for _ in range(int(rng.integers(20, 60)))]
    email, merchant = ids.email(), _merchant(rng)
    bin_ = BINS[rng.integers(0, len(BINS))]
    return [
        _row(t, ids.card(), bin_, _attack_amount(rng), pool[rng.integers(0, len(pool))],
             email, merchant, rng.random() < 0.85, 1, "attack_rotating_ip", campaign)
        for t in times
    ]


def _attack_rotating_identity(rng, ids: _Ids, campaign: str) -> list[dict]:
    """A fresh email per attempt and IPs from a proxy pool - only the BIN and the target
    merchant tie the attempts together, so per-IP and per-email windows see nothing."""
    n = int(rng.integers(30, 101))
    start = rng.uniform(0, HORIZON_SECONDS - 900)
    times = np.sort(start + rng.uniform(0, rng.uniform(120, 600), size=n))
    pool = [ids.ip() for _ in range(int(rng.integers(20, 60)))]
    merchant = _merchant(rng)
    bin_ = BINS[rng.integers(0, len(BINS))]
    return [
        _row(t, ids.card(), bin_, _attack_amount(rng), pool[rng.integers(0, len(pool))],
             ids.email(), merchant, rng.random() < 0.85, 1, "attack_rotating_identity",
             campaign)
        for t in times
    ]


ATTACK_KINDS = [
    (_attack_burst, 0.45),
    (_attack_slow, 0.2),
    (_attack_rotating_ip, 0.15),
    (_attack_rotating_identity, 0.2),
]


def generate(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = _Ids(rng)

    attack_budget = int(n * ATTACK_SHARE)
    flash_events = int(n * FLASH_SALE_SHARE)
    # Retries add roughly decline_rate * retry_prob * 2 extra rows per purchase.
    retry_overhead = 1 + NORMAL_DECLINE_RATE * RETRY_AFTER_DECLINE_PROB * 2
    n_purchases = int((n - attack_budget - flash_events * retry_overhead) / retry_overhead)

    customers = _make_customers(rng, ids, n_customers=max(n_purchases // 6, 50))

    rows: list[dict] = []
    for _ in range(n_purchases):
        customer = customers[rng.integers(0, len(customers))]
        rows.extend(
            _normal_purchase(rng, customer, _normal_ts(rng), _merchant(rng), customer["scenario"])
        )

    rows.extend(_flash_sale(rng, ids, customers, flash_events))

    kinds, weights = zip(*ATTACK_KINDS, strict=True)
    p = np.array(weights) / sum(weights)
    attack_rows = 0
    campaign_no = 0
    while attack_rows < attack_budget:
        campaign_no += 1
        kind = kinds[rng.choice(len(kinds), p=p)]
        new = kind(rng, ids, f"c{campaign_no:04d}")
        rows.extend(new)
        attack_rows += len(new)

    df = pd.DataFrame(rows)
    df = df[df["ts"] < HORIZON_SECONDS].sort_values("ts", kind="stable").reset_index(drop=True)
    df.insert(0, "event_id", [f"evt_{i:07d}" for i in range(len(df))])
    return df[COLUMNS]


def summarize(df: pd.DataFrame) -> str:
    declined = df["outcome"] == "declined"
    lines = [
        f"rows: {len(df):,}",
        f"attack share: {df['is_attack'].mean():.2%} "
        f"({df['is_attack'].sum():,} rows, {df.loc[df.is_attack == 1, 'campaign_id'].nunique()}"
        " campaigns)",
        f"decline rate - normal: {declined[df.is_attack == 0].mean():.1%}, "
        f"attack: {declined[df.is_attack == 1].mean():.1%}",
        "rows per scenario:",
    ]
    for scenario, count in df["scenario"].value_counts().sort_index().items():
        lines.append(f"  {scenario:<20} {count:>7,}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("ml/data/transactions.csv"))
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(summarize(df))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
