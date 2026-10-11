import json

import joblib
import pytest

from ml.dataset import MODEL_FEATURES, build_matrix, time_split
from ml.generate_data import generate
from ml.train import train


@pytest.fixture(scope="module")
def df():
    # Big enough that every slice holds attack campaigns (4000 rows has none in val).
    return generate(n=20_000, seed=7)


def test_time_split_is_disjoint_ordered_and_complete(df):
    split = time_split(df)
    counts = split["train"].astype(int) + split["val"].astype(int) + split["test"].astype(int)
    assert (counts == 1).all()
    ts = df["ts"]
    assert ts[split["train"]].max() < ts[split["val"]].min()
    assert ts[split["val"]].max() < ts[split["test"]].min()


def test_build_matrix_columns(df):
    X, y = build_matrix(df)
    assert list(X.columns) == MODEL_FEATURES
    assert len(X) == len(y) == len(df)


def test_train_saves_usable_artifacts(df, tmp_path):
    trained = train(df, out_dir=tmp_path)
    assert set(trained) == {"logreg", "gbm"}

    X, _ = build_matrix(df)
    test_rows = X[time_split(df)["test"]]
    for name in trained:
        meta = json.loads((tmp_path / f"{name}.json").read_text())
        assert meta["features"] == MODEL_FEATURES
        assert 0.0 <= meta["threshold"] <= 1.0

        model = joblib.load(tmp_path / f"{name}.joblib")
        proba = model.predict_proba(test_rows[meta["features"]])[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all()


def test_train_refuses_val_slice_without_attacks(tmp_path):
    with pytest.raises(ValueError, match="no attack events in the val slice"):
        train(generate(n=4000, seed=7), out_dir=tmp_path)
