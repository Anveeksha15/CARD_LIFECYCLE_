"""Replay generated traffic through the Redis features and compare with the offline ones.

Usage: python -m ml.check_parity [--n 20000] [--seed 42] [--fake]

Without --fake it uses the real Redis at settings.redis_url (`docker compose up -d redis`),
under a unique key prefix that it deletes afterwards - it never flushes the database.
"""

import argparse
import time
import uuid

import numpy as np
import pandas as pd
import redis

from common.config import settings
from ml.generate_data import generate
from ml.offline_features import compute_features
from services.fraud.features import FEATURE_NAMES, Tx, observe, record_outcome


def replay(r: redis.Redis, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Feed events in ts order exactly as the gateway would: score, then report the outcome."""
    rows = []
    for e in df.itertuples(index=False):
        tx = Tx(e.event_id, e.card_token, e.bin, e.ip, e.email)
        rows.append(observe(r, tx, e.ts, prefix=prefix))
        record_outcome(r, tx, declined=e.outcome == "declined", prefix=prefix)
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=df.index)


def mismatches(offline: pd.DataFrame, online: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows where any feature differs between the two paths."""
    close = np.isclose(offline[FEATURE_NAMES], online[FEATURE_NAMES], rtol=1e-9, atol=1e-12)
    return pd.Series(~close.all(axis=1), index=offline.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fake", action="store_true", help="use fakeredis instead of Redis")
    args = parser.parse_args()

    if args.fake:
        import fakeredis

        r = fakeredis.FakeRedis(decode_responses=True)
    else:
        r = redis.from_url(settings.redis_url, decode_responses=True)

    df = generate(args.n, args.seed)
    offline = compute_features(df)

    prefix = f"parity:{uuid.uuid4().hex[:8]}"
    start = time.perf_counter()
    try:
        online = replay(r, df, prefix)
    finally:
        elapsed = time.perf_counter() - start
        for key in r.scan_iter(f"{prefix}:*"):
            r.delete(key)

    bad = mismatches(offline, online)
    print(f"events: {len(df):,}  backend: {'fakeredis' if args.fake else settings.redis_url}")
    print(f"replay: {elapsed:.2f}s ({elapsed / len(df) * 1000:.3f} ms per event incl. outcome)")
    print(f"mismatching rows: {int(bad.sum())}")
    if bad.any():
        print(offline[bad].head().join(online[bad].head(), rsuffix="_redis").T)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
