"""Train the fraud models on the time-based split and save them with their metadata.

Usage: python -m ml.train [--data ml/data/transactions.csv] [--out ml/artifacts]

Each model is fit on the train slice; its decision threshold is the one that maximizes F1
on the validation slice. The test slice is never touched here - see ml/evaluate.py.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml.dataset import (
    DEFAULT_DATA,
    MODEL_FEATURES,
    TEST_START,
    VAL_START,
    build_matrix,
    load,
    time_split,
)

DEFAULT_OUT = Path("ml/artifacts")


def make_models() -> dict:
    return {
        # Counts are heavy-tailed; log1p keeps one 200-attempt burst from dominating the fit.
        "logreg": make_pipeline(
            FunctionTransformer(np.log1p),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0),
        ),
        "gbm": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=200, learning_rate=0.1, random_state=0
        ),
    }


def best_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) maximizing F1 on the given slice."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall point has no threshold attached.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def train(df: pd.DataFrame, out_dir: Path = DEFAULT_OUT) -> dict[str, dict]:
    X, y = build_matrix(df)
    split = time_split(df)
    for name in ("train", "val"):
        if not y[split[name]].any():
            raise ValueError(
                f"no attack events in the {name} slice - generate more data (--n) so the "
                "model can be fit and its threshold tuned"
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    trained = {}
    for name, model in make_models().items():
        model.fit(X[split["train"]], y[split["train"]])
        val_scores = model.predict_proba(X[split["val"]])[:, 1]
        threshold, val_f1 = best_f1_threshold(y[split["val"]], val_scores)

        meta = {
            "model": name,
            "features": MODEL_FEATURES,
            "threshold": threshold,
            "val_f1": val_f1,
            "split": {"val_start": VAL_START, "test_start": TEST_START},
            "train_rows": int(split["train"].sum()),
            "train_attack_rows": int(y[split["train"]].sum()),
            "sklearn_version": sklearn.__version__,
        }
        joblib.dump(model, out_dir / f"{name}.joblib")
        (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        trained[name] = meta
    return trained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for name, meta in train(load(args.data), args.out).items():
        print(f"{name}: threshold={meta['threshold']:.4f}  val F1={meta['val_f1']:.3f}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
