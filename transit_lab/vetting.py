"""Vetting classifiers: the dual-view CNN and the Random Forest baseline.

Both consume the SAME inputs (AstroNet global + local views), so any performance
gap is attributable to model class, not feature engineering. The RF takes the two
views concatenated into one 2202-dim vector.

CNN architecture rationale: one fixed resolution cannot serve both
jobs. Enough resolution to resolve transit shape leaves too short a window to see
context; enough window for context leaves the transit spanning only a few bins.
Two views at two scales, two conv stacks, concatenated before the head.

Data: AstroNet's pre-computed views (scripts/convert_tfrecords.py -> .npz) on
AstroNet's released split. That split is by TCE, so stars leak across splits;
test.npz carries an `unseen_star` mask, and every test metric is reported on both
the full set and the unseen-star subset (see DESIGN.md).
"""

from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

TCEs = dict[str, np.ndarray]


def load_views(path: str | Path) -> TCEs:
    # Load one split written by convert_tfrecords.py fully into memory
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def rf_features(tces: TCEs) -> np.ndarray:
    # (N, 2202) matrix: each row is one TCE's global view followed by its local view
    return np.concatenate([tces["global_view"], tces["local_view"]], axis=1)


def train_rf(train: TCEs, n_trees: int = 500, seed: int = 0) -> RandomForestClassifier:
    # Fit the RF on the train split. Defaults are sklearn's apart from tree count.
    # Deterministic given `seed`, regardless of n_jobs, which controlls parallel processing.
    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    rf.fit(rf_features(train), train["label"])
    return rf


def planet_scores(rf: RandomForestClassifier, tces: TCEs) -> np.ndarray:
    """Planet score per TCE: the mean over trees of the planet fraction in its leaf.
    Can be computes as for one planet, and for many planets.
    """
    if list(rf.classes_) != [0, 1]:
        raise ValueError(f"expected classes [0, 1], got {rf.classes_}")
    proba = rf.predict_proba(rf_features(tces))
    return proba[:, 1]


def vetting_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    # Threshold-free ranking quality: ROC AUC and PR-AUC (average precision)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "n": len(labels),
        "n_planets": int(labels.sum()),
    }


def evaluate(scores: np.ndarray, tces: TCEs) -> dict[str, dict[str, float]]:
    # Metrics on the whole split, plus on the unseen-star subset when it is flagged.
    results = {"all": vetting_metrics(tces["label"], scores)}
    if "unseen_star" in tces:
        m = tces["unseen_star"]
        results["unseen_star"] = vetting_metrics(tces["label"][m], scores[m])
    return results


class DualViewCNN:
    """Two 1-D conv stacks (global, local) -> concatenate -> dense head -> sigmoid."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError