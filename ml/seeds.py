"""Re-run the whole pipeline on several dataset seeds to see how stable the results are.

Usage: python -m ml.seeds [--seeds 1 2 3 4 5] [--n 100000]

The test slice of one dataset holds only ~15 attack campaigns, so a single seed's numbers
can move by several points. This regenerates the data, retrains and re-evaluates per seed
(about a minute each) and reports mean, standard deviation and min-max per method.
"""

import argparse
import statistics
import tempfile
from pathlib import Path

from ml.evaluate import evaluate
from ml.generate_data import generate
from ml.train import train

RESULTS = Path("ml/results/seeds.md")
METRICS = ["precision", "recall", "f1", "fpr", "campaign_recall"]


def run(seeds: list[int], n: int) -> dict[str, dict[str, list[float]]]:
    collected: dict[str, dict[str, list[float]]] = {}
    for seed in seeds:
        df = generate(n, seed)
        with tempfile.TemporaryDirectory() as tmp:
            train(df, out_dir=Path(tmp))
            results = evaluate(df, Path(tmp))
        for method, m in results["metrics"].items():
            for metric in METRICS:
                collected.setdefault(method, {}).setdefault(metric, []).append(m[metric])
        gbm = results["metrics"]["gbm"]
        print(f"seed {seed}: gbm F1={gbm['f1']:.3f} FPR={gbm['fpr']:.4f}, "
              f"rules F1={results['metrics']['rules']['f1']:.3f}, "
              f"{results['test_campaigns']} test campaigns")
    return collected


def to_markdown(collected: dict, seeds: list[int], n: int) -> str:
    lines = [
        "# Seed robustness",
        "",
        f"`python -m ml.seeds --seeds {' '.join(map(str, seeds))} --n {n}`: each seed "
        "regenerates the synthetic data, retrains, and evaluates on its own test slice.",
        "Cells are mean ± standard deviation (min-max).",
        "",
        "| method | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for method, metrics in collected.items():
        cells = []
        for metric in METRICS:
            v = metrics[metric]
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            digits = 4 if metric == "fpr" else 3
            cells.append(f"{statistics.mean(v):.{digits}f} ± {sd:.{digits}f} "
                         f"({min(v):.{digits}f}-{max(v):.{digits}f})")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n", type=int, default=100_000)
    args = parser.parse_args()

    markdown = to_markdown(run(args.seeds, args.n), args.seeds, args.n)
    RESULTS.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
