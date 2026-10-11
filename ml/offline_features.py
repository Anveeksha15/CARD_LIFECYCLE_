"""Sliding-window features computed offline from the transaction CSV.

Each event's features use only what a real-time scorer would know at that moment: all
earlier events, plus the current event's own identifiers (ip, email, card, bin). The
current event's *outcome* is never used - at scoring time the payment hasn't been
authorized yet. The Redis version (services/fraud/features.py) follows the same rules,
including the window boundary: an event counts if `ts > now - window`.
"""

from collections import defaultdict, deque

import pandas as pd

from services.fraud.features import (
    BIN_WINDOW,
    EMAIL_WINDOW,
    FEATURE_NAMES,
    IP_LONG_WINDOW,
    IP_SHORT_WINDOW,
)


def _evict(window: deque, now: float, seconds: float) -> None:
    while window and window[0][0] <= now - seconds:
        window.popleft()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row of features per event, aligned with `df` (which must be sorted by ts)."""
    ip_events: dict[str, deque] = defaultdict(deque)  # (ts, card_token, declined)
    email_events: dict[str, deque] = defaultdict(deque)  # (ts,)
    bin_events: dict[str, deque] = defaultdict(deque)  # (ts,)

    out = {name: [] for name in FEATURE_NAMES}
    for ts, card, bin_, ip, email, outcome in zip(
        df["ts"], df["card_token"], df["bin"], df["ip"], df["email"], df["outcome"], strict=True
    ):
        ip_win = ip_events[ip]
        _evict(ip_win, ts, IP_LONG_WINDOW)
        recent = [e for e in ip_win if e[0] > ts - IP_SHORT_WINDOW]
        prior_failed_5m = sum(e[2] for e in ip_win)

        out["attempts_per_ip_60s"].append(len(recent) + 1)
        out["failed_per_ip_60s"].append(sum(e[2] for e in recent))
        out["distinct_cards_per_ip_5m"].append(len({e[1] for e in ip_win} | {card}))
        out["decline_rate_ip_5m"].append(prior_failed_5m / len(ip_win) if ip_win else 0.0)

        email_win = email_events[email]
        _evict(email_win, ts, EMAIL_WINDOW)
        out["attempts_per_email_60s"].append(len(email_win) + 1)

        bin_win = bin_events[bin_]
        _evict(bin_win, ts, BIN_WINDOW)
        out["attempts_per_bin_60s"].append(len(bin_win) + 1)

        # The outcome becomes known only after scoring, so record the event afterwards.
        ip_win.append((ts, card, outcome == "declined"))
        email_win.append((ts,))
        bin_win.append((ts,))

    return pd.DataFrame(out, index=df.index)
