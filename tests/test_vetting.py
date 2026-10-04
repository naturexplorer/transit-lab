"""Tests for the vetting classifiers (stage 7), on synthetic views."""

import numpy as np
from transit_lab.vetting import evaluate, planet_scores, rf_features, train_rf, train_cnn, cnn_scores, cnn_tensors, vetting_metrics


def make_split(n: int, seed: int, with_mask: bool = False) -> dict[str, np.ndarray]:
    # Normal-dist generated views; planets get a box-shaped dip at the local view's centre
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n).astype(np.int8)
    global_view = rng.normal(0, 1, (n, 2001)).astype(np.float32)
    local_view = rng.normal(0, 1, (n, 201)).astype(np.float32)
    local_view[:, 95:106] -= 3 * label[:, None]
    split = {"global_view": global_view, "local_view": local_view, "label": label}
    if with_mask:
        split["star_disjoint"] = rng.random(n) < 0.5
    return split


def test_rf_features_are_global_then_local():
    split = make_split(4, 0)
    x = rf_features(split)
    assert x.shape == (4, 2202)
    np.testing.assert_array_equal(x[:, :2001], split["global_view"])
    np.testing.assert_array_equal(x[:, 2001:], split["local_view"])

def test_cnn_tensors_produces_right_shape():
    split = make_split(4, 0)
    dataset = cnn_tensors(split)
    (g, l, y) = dataset.tensors
    assert(g.shape == (4, 1, 2001))
    assert(l.shape == (4, 1, 201))
    assert(y.shape == (4,))

def test_rf_separates_planets_from_noise():
    rf = train_rf(make_split(300, 0), n_trees=50)
    held_out = make_split(200, 1)
    assert vetting_metrics(held_out["label"], planet_scores(rf, held_out))["auc"] > 0.95


def test_cnn_separates_planets_from_noise():
    cnn, _ = train_cnn(make_split(300, 0), epochs=10)
    held_out = make_split(200, 1)
    assert vetting_metrics(held_out["label"], cnn_scores(cnn, held_out))["auc"] > 0.95


def test_rf_is_deterministic_given_seed():
    train, held_out = make_split(100, 0), make_split(50, 1)
    first = planet_scores(train_rf(train, n_trees=20, seed=7), held_out)
    second = planet_scores(train_rf(train, n_trees=20, seed=7), held_out)
    np.testing.assert_array_equal(first, second)


def test_metrics_on_perfect_and_inverted_rankings():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    perfect = vetting_metrics(labels, scores)
    assert perfect["auc"] == 1.0 and perfect["pr_auc"] == 1.0
    assert vetting_metrics(labels, -scores)["auc"] == 0.0


def test_star_disjoint_subset_reported_only_when_flagged():
    scores = np.random.default_rng(0).random(100)
    assert evaluate(scores, make_split(100, 0)).keys() == {"all"}

    split = make_split(100, 0, with_mask=True)
    results = evaluate(scores, split)
    assert results.keys() == {"all", "star_disjoint"}
    assert results["star_disjoint"]["n"] == split["star_disjoint"].sum()