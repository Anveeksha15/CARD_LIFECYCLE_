"""Classification metrics for imbalanced fraud data. Accuracy is deliberately absent: with
~3.5% attacks, always answering "normal" already scores ~96.5%."""

from collections.abc import Sequence


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn)."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t and p:
            tp += 1
        elif p:
            fp += 1
        elif t:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def precision(y_true, y_pred) -> float:
    tp, fp, _, _ = confusion(y_true, y_pred)
    return _ratio(tp, tp + fp)


def recall(y_true, y_pred) -> float:
    tp, _, fn, _ = confusion(y_true, y_pred)
    return _ratio(tp, tp + fn)


def f1(y_true, y_pred) -> float:
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return _ratio(2 * p * r, p + r) if p + r else 0.0


def false_positive_rate(y_true, y_pred) -> float:
    """Share of normal events that got flagged - the cost paid by real customers."""
    _, fp, _, tn = confusion(y_true, y_pred)
    return _ratio(fp, fp + tn)


def campaign_recall(campaign_ids: Sequence[str], y_pred: Sequence[int]) -> float:
    """Share of attack campaigns with at least one flagged event. Empty ids are normal rows."""
    caught: dict[str, bool] = {}
    for cid, p in zip(campaign_ids, y_pred, strict=True):
        if cid:
            caught[cid] = caught.get(cid, False) or bool(p)
    return _ratio(sum(caught.values()), len(caught))


def report(y_true, y_pred, campaign_ids) -> dict:
    tp, fp, fn, tn = confusion(y_true, y_pred)
    return {
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1(y_true, y_pred),
        "fpr": false_positive_rate(y_true, y_pred),
        "campaign_recall": campaign_recall(campaign_ids, y_pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
