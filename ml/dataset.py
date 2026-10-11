"""Loading, time-based splitting and model inputs shared by baseline, train and evaluate.

The split is by time, never random: a random split would put events from the same attack
campaign on both sides, so the model would be graded on patterns it already saw.
"""

from pathlib import Path

import pandas as pd

from ml.generate_data import HORIZON_SECONDS
from ml.offline_features import compute_features
from services.fraud.features import FEATURE_NAMES

DEFAULT_DATA = Path("ml/data/transactions.csv")

VAL_START = 0.6 * HORIZON_SECONDS  # train: [0, 60%) of the timeline
TEST_START = 0.7 * HORIZON_SECONDS  # val: [60%, 70%) picks thresholds; test: [70%, 100%]

# amount_cents comes straight from the payment request, so it needs no Redis window.
MODEL_FEATURES = FEATURE_NAMES + ["amount_cents"]


def load(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"bin": str, "campaign_id": str})
    df["campaign_id"] = df["campaign_id"].fillna("")
    return df


def time_split(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "train": df["ts"] < VAL_START,
        "val": (df["ts"] >= VAL_START) & (df["ts"] < TEST_START),
        "test": df["ts"] >= TEST_START,
    }


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features are computed over the whole stream in time order, then split - each
    event's windows only ever look backwards, so this leaks nothing across the split."""
    X = compute_features(df)
    X["amount_cents"] = df["amount_cents"]
    return X[MODEL_FEATURES], df["is_attack"]
