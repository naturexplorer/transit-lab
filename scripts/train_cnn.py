"""Train the CNN on AstroNet's train split and report vetting metrics.

Run from the repo root:  python scripts/train_cnn.py            (validation metrics for seed 0)
                         python scripts/train_cnn.py --seeds k  (validation metrics for seeds 0..k-1)
"""

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from transit_lab.vetting import evaluate, load_views, train_cnn, cnn_scores


DATA_DIR = Path("astronet_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the CNN vetter.")
    parser.add_argument("--seeds", type=int, default=1,
                        help="train one CNN per seed 0..N-1 and report mean ± std")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
 
    # Load once: the data doesn't change between seeds, only the CNN parameters do.
    train = load_views(DATA_DIR / "train.npz")
    MODEL_DIR = Path("models") 
    MODEL_DIR.mkdir(exist_ok=True)      
    val = load_views(DATA_DIR / f"val.npz")

    #           subset, list of results for each run
    results: dict[str, list[dict[str, float]]] = {}
    for seed in range(args.seeds):
        start = time.perf_counter()
        cnn, best_epoch = train_cnn(train, val, seed=seed)
        torch.save({"state_dict": cnn.state_dict(), "seed": seed, "best_epoch": best_epoch},
               MODEL_DIR / f"cnn_seed{seed}.pt")
        for subset, m in evaluate(cnn_scores(cnn, val), val).items():
            results.setdefault(subset, []).append(m)
            print(f"seed {seed}  val/{subset:11s}  "
                  f"AUC={m['auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
        print(f"seed {seed}  done in {time.perf_counter() - start:.0f} s")
 
    print(f"\nsummary over {args.seeds} seed(s):")
    for subset, runs in results.items():
        n, n_pc = runs[0]["n"], runs[0]["n_planets"]  # same rows for every seed
        line = f"val/{subset:11s}"
        for metric, label in (("auc", "AUC"), ("pr_auc", "PR-AUC")):
            values = np.array([r[metric] for r in runs])
            # Sample std (ddof=1): the seeds are a sample of all possible seeds.
            spread = f" ± {values.std(ddof=1):.4f}" if len(values) > 1 else ""
            line += f"  {label}={values.mean():.4f}{spread}"
        print(f"{line}  n={n}  planets={n_pc}  (PR-AUC of random ranking: {n_pc / n:.3f})")
 

if __name__ == "__main__":
    main()