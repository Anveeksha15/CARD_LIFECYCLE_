"""Loads a model saved by ml/train.py and scores one payment at a time.

The column order and block threshold come from the artifact's JSON metadata, so the
service can't silently feed features in a different order than the model was trained on.
"""

import json
from pathlib import Path

import joblib
import pandas as pd


class FraudModel:
    def __init__(self, model, meta: dict):
        self._model = model
        self.meta = meta

    @classmethod
    def load(cls, path_prefix: str | Path) -> "FraudModel":
        prefix = Path(path_prefix)
        model_file, meta_file = prefix.with_suffix(".joblib"), prefix.with_suffix(".json")
        missing = [str(f) for f in (model_file, meta_file) if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"fraud model artifact not found: {', '.join(missing)} - "
                "run `python -m ml.generate_data && python -m ml.train` first"
            )
        return cls(joblib.load(model_file), json.loads(meta_file.read_text()))

    @property
    def name(self) -> str:
        return self.meta["model"]

    @property
    def threshold(self) -> float:
        return self.meta["threshold"]

    def predict(self, features: dict, amount_cents: int) -> float:
        row = {**features, "amount_cents": amount_cents}
        X = pd.DataFrame([[row[name] for name in self.meta["features"]]],
                         columns=self.meta["features"])
        return float(self._model.predict_proba(X)[0, 1])
