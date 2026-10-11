"""Training/serving parity: the Redis features must equal the offline ones, row for row."""

import fakeredis
import numpy as np
import pandas as pd

from ml.check_parity import mismatches, replay
from ml.generate_data import _attack_burst, _Ids, generate
from ml.offline_features import compute_features


def _sample_with_burst() -> pd.DataFrame:
    # A small sample may contain no fast burst at all, so add one to make sure the
    # windows actually fill up rather than comparing mostly all-ones features.
    rng = np.random.default_rng(0)
    burst = pd.DataFrame(_attack_burst(rng, _Ids(rng), "c_test"))
    burst["event_id"] = [f"burst_{i}" for i in range(len(burst))]
    df = pd.concat([generate(n=3000, seed=7), burst], ignore_index=True)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def test_redis_features_match_offline_features():
    df = _sample_with_burst()
    r = fakeredis.FakeRedis(decode_responses=True)

    offline = compute_features(df)
    online = replay(r, df, prefix="parity")

    bad = mismatches(offline, online)
    assert not bad.any(), offline[bad].head().join(online[bad].head(), rsuffix="_redis")
    assert offline["attempts_per_ip_60s"].max() > 10
    assert offline["decline_rate_ip_5m"].max() > 0.5
