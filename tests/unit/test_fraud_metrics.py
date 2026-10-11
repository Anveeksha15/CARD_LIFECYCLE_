import pytest

from ml.metrics import (
    campaign_recall,
    confusion,
    f1,
    false_positive_rate,
    precision,
    recall,
)

# 3 attacks, 5 normal: tp=2, fn=1, fp=1, tn=4
Y_TRUE = [1, 1, 1, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 0, 1, 0, 0, 0, 0]


def test_confusion():
    assert confusion(Y_TRUE, Y_PRED) == (2, 1, 1, 4)


def test_scores():
    assert precision(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert recall(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert f1(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert false_positive_rate(Y_TRUE, Y_PRED) == pytest.approx(1 / 5)


def test_no_predictions_does_not_divide_by_zero():
    assert precision([1, 0], [0, 0]) == 0.0
    assert f1([1, 0], [0, 0]) == 0.0


def test_campaign_recall():
    campaigns = ["c1", "c1", "c2", "", ""]
    preds = [0, 1, 0, 1, 0]
    assert campaign_recall(campaigns, preds) == 0.5
