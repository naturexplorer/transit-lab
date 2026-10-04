"""Train the Random Forest on AstroNet's train split and report vetting metrics.

Run from the repo root:  python scripts/train_rf.py             (validation metrics for seed 0)
                         python scripts/train_rf.py --seeds k  (validation metrics for seeds 0..k-1)
                         python scripts/train_rf.py --test      (test metrics for seed 0)

The --test is only for the final, frozen model: every decision made after looking at
test turns it into a second validation set.
"""

import argparse
import time
from pathlib import Path
import numpy as np
from transit_lab.vetting import evaluate, load_views, rf_scores, train_rf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "astronet_data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the RF vetter.")
    parser.add_argument("--seeds", type=int, default=1,
                        help="train one forest per seed 0..N-1 and report mean ± std")
    parser.add_argument("--test", action="store_true", help="evaluate on test, not val")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
 
    # Load once: the data doesn't change between seeds, only the forest does.
    train = load_views(DATA_DIR / "train.npz")
    name = "test" if args.test else "val"
    split = load_views(DATA_DIR / f"{name}.npz")
 
    # Keep metrics, not forests: each forest can be hundreds of MB, so it is
    # scored and discarded inside the loop.
    #           subset, list of results for each run
    results: dict[str, list[dict[str, float]]] = {}
    for seed in range(args.seeds):
        start = time.perf_counter()
        rf = train_rf(train, seed=seed)
        for subset, m in evaluate(rf_scores(rf, split), split).items():
            results.setdefault(subset, []).append(m)
            print(f"seed {seed}  {name}/{subset:11s}  "
                  f"AUC={m['auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
        print(f"seed {seed}  done in {time.perf_counter() - start:.0f} s")
 
    print(f"\nsummary over {args.seeds} seed(s):")
    for subset, runs in results.items():
        n, n_pc = runs[0]["n"], runs[0]["n_planets"]  # same rows for every seed
        line = f"{name}/{subset:11s}"
        for metric, label in (("auc", "AUC"), ("pr_auc", "PR-AUC")):
            values = np.array([r[metric] for r in runs])
            # Sample std (ddof=1): the seeds are a sample of all possible seeds.
            spread = f" ± {values.std(ddof=1):.4f}" if len(values) > 1 else ""
            line += f"  {label}={values.mean():.4f}{spread}"
        print(f"{line}  n={n}  planets={n_pc}  (PR-AUC of random ranking: {n_pc / n:.3f})")
 

if __name__ == "__main__":
    main()