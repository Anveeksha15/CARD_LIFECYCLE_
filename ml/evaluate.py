"""Compare the rules baseline with the trained models on the held-out test slice.

Usage: python -m ml.evaluate [--data ml/data/transactions.csv] [--artifacts ml/artifacts]
Run python -m ml.train first.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from ml.baseline import rules_predictions
from ml.dataset import DEFAULT_DATA, build_matrix, load, time_split
from ml.metrics import report
from ml.train import DEFAULT_OUT

RESULTS_DIR = Path("ml/results")
MODEL_NAMES = ["logreg", "gbm"]


def load_model(artifacts: Path, name: str):
    meta = json.loads((artifacts / f"{name}.json").read_text())
    return joblib.load(artifacts / f"{name}.joblib"), meta


def evaluate(df: pd.DataFrame, artifacts: Path) -> dict:
    X, y = build_matrix(df)
    test_mask = time_split(df)["test"]
    test, X_test, y_test = df[test_mask], X[test_mask], y[test_mask]

    preds = {"rules": rules_predictions(X_test)}
    scores, models = {}, {}
    for name in MODEL_NAMES:
        model, meta = load_model(artifacts, name)
        models[name] = model
        scores[name] = model.predict_proba(X_test[meta["features"]])[:, 1]
        preds[name] = (scores[name] >= meta["threshold"]).astype(int).tolist()

    table = {}
    for name, p in preds.items():
        table[name] = report(y_test, p, test["campaign_id"])
        table[name]["pr_auc"] = (
            float(average_precision_score(y_test, scores[name])) if name in scores else None
        )

    flagged = test.assign(**{name: p for name, p in preds.items()})
    by_scenario = flagged.groupby("scenario")[list(preds)].mean().round(4)

    best = max(MODEL_NAMES, key=lambda n: table[n]["f1"])
    attacks = flagged[flagged["is_attack"] == 1]
    campaigns = attacks.groupby("campaign_id").agg(
        scenario=("scenario", "first"), size=(best, "size"), caught=(best, "mean")
    )
    normals = flagged[(flagged["is_attack"] == 0) & (flagged[best] == 1)]

    importance = permutation_importance(
        models["gbm"], X_test, y_test, scoring="average_precision", n_repeats=5, random_state=0
    )

    return {
        "test_rows": int(len(test)),
        "test_attack_rows": int(y_test.sum()),
        "test_campaigns": int(attacks["campaign_id"].nunique()),
        "metrics": table,
        "flag_rate_by_scenario": by_scenario.to_dict(orient="index"),
        "best_model": best,
        "worst_campaigns": campaigns.sort_values("caught").head(8).reset_index()
        .to_dict(orient="records"),
        "false_positives_by_scenario": normals["scenario"].value_counts().to_dict(),
        "top_false_positive_ips": normals["ip"].value_counts().head(5).to_dict(),
        "gbm_permutation_importance": dict(
            sorted(
                zip(X_test.columns, importance.importances_mean.round(4).tolist(), strict=True),
                key=lambda kv: -kv[1],
            )
        ),
    }


def to_markdown(r: dict) -> str:
    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    lines = [
        "# Rules vs models (test slice)",
        "",
        f"Synthetic data only. Test slice = last 30% of the timeline: {r['test_rows']:,} events,"
        f" {r['test_attack_rows']:,} attack events in {r['test_campaigns']} campaigns.",
        "Model thresholds were chosen on the validation slice (60-70%), never on test.",
        "",
        "| method | precision | recall | F1 | FPR (normal) | campaign recall | PR-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in r["metrics"].items():
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['fpr']:.4f} | {m['campaign_recall']:.3f} | {fmt(m['pr_auc'], '.3f')} |"
        )

    methods = list(r["metrics"])
    lines += [
        "",
        "## Share flagged by scenario",
        "",
        "Attack rows: higher is better. Normal rows: lower is better.",
        "",
        "| scenario | " + " | ".join(methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for scenario, rates in r["flag_rate_by_scenario"].items():
        lines.append(f"| {scenario} | " + " | ".join(f"{rates[m]:.3f}" for m in methods) + " |")

    lines += [
        "",
        f"## Error analysis ({r['best_model']})",
        "",
        "Least-caught attack campaigns:",
        "",
        "| campaign | scenario | events | share caught |",
        "|---|---|---|---|",
    ]
    for c in r["worst_campaigns"]:
        lines.append(f"| {c['campaign_id']} | {c['scenario']} | {c['size']} | {c['caught']:.2f} |")
    lines += ["", "False positives by scenario:", ""]
    lines += [f"- {s}: {n}" for s, n in r["false_positives_by_scenario"].items()] or ["- none"]
    lines += ["", "Most-flagged normal IPs:", ""]
    lines += [f"- {ip}: {n}" for ip, n in r["top_false_positive_ips"].items()] or ["- none"]

    lines += [
        "",
        "## gbm permutation importance (drop in PR-AUC on test)",
        "",
        "| feature | importance |",
        "|---|---|",
    ]
    lines += [f"| {f} | {v:.4f} |" for f, v in r["gbm_permutation_importance"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = evaluate(load(args.data), args.artifacts)
    markdown = to_markdown(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (RESULTS_DIR / "results.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
