"""Real-time sliding-window features in Redis.

This module is the single definition of the feature set: ml/offline_features.py imports
the names and windows from here, and tests/unit/test_feature_parity.py replays the same
event stream through both paths and requires identical numbers.

Each payment touches Redis twice:
  1. observe() at score time - records the attempt and returns its features.
  2. record_outcome() after authorization - records a decline, if there was one.
So, like the offline path, an event's features never include its own outcome, and an
attempt that gets blocked still counts against its IP/email/BIN.

Windows keep events with `ts > now - window`: ZREMRANGEBYSCORE's inclusive max drops an
event exactly `window` seconds old, and short counts use an exclusive "(" lower bound.
"""

from dataclasses import dataclass

import redis

IP_SHORT_WINDOW = 60
IP_LONG_WINDOW = 300
EMAIL_WINDOW = 60
BIN_WINDOW = 60

FEATURE_NAMES = [
    "attempts_per_ip_60s",
    "failed_per_ip_60s",
    "distinct_cards_per_ip_5m",
    "decline_rate_ip_5m",
    "attempts_per_email_60s",
    "attempts_per_bin_60s",
]


@dataclass(frozen=True)
class Tx:
    event_id: str
    card_token: str
    bin: str
    ip: str
    email: str


def _keys(tx: Tx, prefix: str) -> dict[str, str]:
    return {
        "ip_attempts": f"{prefix}:ip:{tx.ip}:attempts",
        "ip_failed": f"{prefix}:ip:{tx.ip}:failed",
        "ip_cards": f"{prefix}:ip:{tx.ip}:cards",
        "email": f"{prefix}:email:{tx.email}:attempts",
        "bin": f"{prefix}:bin:{tx.bin}:attempts",
    }


def _since(now: float, window: int) -> str:
    # Exclusive lower bound; repr() round-trips the float exactly through Redis.
    return f"({now - window!r}"


def observe(r: redis.Redis, tx: Tx, now: float, prefix: str = "fraud") -> dict:
    """Record this attempt and return its features, in one MULTI round trip."""
    k = _keys(tx, prefix)
    pipe = r.pipeline(transaction=True)

    for key in (k["ip_attempts"], k["ip_failed"], k["ip_cards"]):
        pipe.zremrangebyscore(key, "-inf", now - IP_LONG_WINDOW)
    pipe.zremrangebyscore(k["email"], "-inf", now - EMAIL_WINDOW)
    pipe.zremrangebyscore(k["bin"], "-inf", now - BIN_WINDOW)

    pipe.zadd(k["ip_attempts"], {tx.event_id: now})
    pipe.zadd(k["ip_cards"], {tx.card_token: now})  # score = last time this card was seen
    pipe.zadd(k["email"], {tx.event_id: now})
    pipe.zadd(k["bin"], {tx.event_id: now})

    first_read = len(pipe)
    pipe.zcount(k["ip_attempts"], _since(now, IP_SHORT_WINDOW), "+inf")
    pipe.zcard(k["ip_attempts"])
    pipe.zcount(k["ip_failed"], _since(now, IP_SHORT_WINDOW), "+inf")
    pipe.zcard(k["ip_failed"])
    pipe.zcard(k["ip_cards"])
    pipe.zcard(k["email"])
    pipe.zcard(k["bin"])

    for key in (k["ip_attempts"], k["ip_failed"], k["ip_cards"]):
        pipe.expire(key, 2 * IP_LONG_WINDOW)
    pipe.expire(k["email"], 2 * EMAIL_WINDOW)
    pipe.expire(k["bin"], 2 * BIN_WINDOW)

    results = pipe.execute()
    attempts_60s, attempts_5m, failed_60s, failed_5m, cards, email, bin_ = results[
        first_read : first_read + 7
    ]

    prior_5m = attempts_5m - 1
    return {
        "attempts_per_ip_60s": attempts_60s,
        "failed_per_ip_60s": failed_60s,
        "distinct_cards_per_ip_5m": cards,
        "decline_rate_ip_5m": failed_5m / prior_5m if prior_5m else 0.0,
        "attempts_per_email_60s": email,
        "attempts_per_bin_60s": bin_,
    }


def record_outcome(r: redis.Redis, tx: Tx, declined: bool, prefix: str = "fraud") -> None:
    """Record the authorization result for an attempt already passed to observe().

    The decline is stamped with the *attempt's* time, not the time this call arrives, so
    it leaves the windows together with its attempt - as in the offline features. If the
    attempt has already expired from the windows (or was never observed), there's nothing
    left for the outcome to count against, so it's dropped."""
    if not declined:
        return
    k = _keys(tx, prefix)
    attempted_at = r.zscore(k["ip_attempts"], tx.event_id)
    if attempted_at is None:
        return
    pipe = r.pipeline(transaction=True)
    pipe.zadd(k["ip_failed"], {tx.event_id: attempted_at})
    pipe.expire(k["ip_failed"], 2 * IP_LONG_WINDOW)
    pipe.execute()
