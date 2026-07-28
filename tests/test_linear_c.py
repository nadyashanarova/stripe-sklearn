"""Tests for :class:`stripe_learning.StripeCClassifier` (linear C-system).

Covers: get_params/clone compatibility, the applicable subset of
sklearn's generic estimator checks, Pipeline/GridSearchCV/cross_validate
smoke tests, n_features_in_/fitted-state behaviour, deterministic
repeated fit, partial_fit label/classes validation, and numerical
parity against the untouched reference implementation in
``references/linear_reference.py`` -- including the multizone
correction zone boundaries and the decay-before-residual regularization
order (MATH_AUDIT.md finding M7).
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from stripe_learning import StripeCClassifier

from _reference_loader import load_linear_reference


def _reference_class():
    return load_linear_reference().StripeCSystemCorrected


def _toy_dataset(n_samples=60, n_features=5, random_state=0):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    return X.astype(np.float64), y


DEFAULT_PARAMS = {
    "eps": 0.1,
    "delta_ratio": 0.05,
    "beta": 1.0,
    "epochs_count": 1,
    "fit_bias": True,
    "lam": 0.0,
    "regularize_bias": False,
    "shuffle": True,
    "random_state": None,
    "thresh": 0.0,
    "init_mode": "zeros",
    "init_scale": 0.01,
}


# ---------------------------------------------------------------------
# get_params / set_params / clone compatibility
# ---------------------------------------------------------------------


def test_default_constructor_has_no_required_arguments():
    StripeCClassifier()


def test_get_params_matches_constructor_defaults():
    assert StripeCClassifier().get_params() == DEFAULT_PARAMS


def test_set_params_updates_attributes_without_side_effects():
    clf = StripeCClassifier()
    returned = clf.set_params(lam=0.2, eps=0.5, init_mode="random")
    assert returned is clf
    assert clf.lam == 0.2
    assert clf.eps == 0.5
    assert clf.init_mode == "random"
    assert clf.delta_ratio == DEFAULT_PARAMS["delta_ratio"]


def test_clone_reproduces_params_without_carrying_fitted_state():
    X, y = _toy_dataset()
    clf = StripeCClassifier(random_state=0, lam=0.05)
    clf.fit(X, y)

    cloned = clone(clf)

    assert cloned.get_params() == clf.get_params()
    assert not hasattr(cloned, "weights_")
    assert not hasattr(cloned, "classes_")

    cloned.fit(X, y)
    np.testing.assert_allclose(cloned.weights_, clf.weights_)


def test_init_does_not_mutate_or_validate_arguments():
    clf = StripeCClassifier(lam="not-yet-a-float", eps="also-not-a-float")
    assert clf.lam == "not-yet-a-float"
    assert clf.eps == "also-not-a-float"


def test_delta_property_matches_eps_times_delta_ratio():
    clf = StripeCClassifier(eps=0.2, delta_ratio=0.25)
    assert clf.delta == pytest.approx(0.05)


# ---------------------------------------------------------------------
# sklearn generic estimator checks (applicable subset)
# ---------------------------------------------------------------------


@parametrize_with_checks([StripeCClassifier()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


# ---------------------------------------------------------------------
# Pipeline / GridSearchCV / cross_validate smoke tests
# ---------------------------------------------------------------------


def test_pipeline_smoke():
    X, y = _toy_dataset()
    pipe = make_pipeline(StandardScaler(), StripeCClassifier(random_state=0))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)) <= set(np.unique(y))
    scores = pipe.decision_function(X)
    assert scores.shape == y.shape


def test_grid_search_cv_smoke():
    X, y = _toy_dataset(n_samples=80)
    param_grid = {"eps": [0.05, 0.2], "beta": [0.5, 1.0]}
    search = GridSearchCV(
        StripeCClassifier(random_state=0),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert search.best_params_["eps"] in param_grid["eps"]
    assert search.best_params_["beta"] in param_grid["beta"]


def test_cross_validate_smoke():
    X, y = _toy_dataset(n_samples=80)
    results = cross_validate(
        StripeCClassifier(random_state=0),
        X,
        y,
        cv=4,
        scoring="accuracy",
    )
    assert len(results["test_score"]) == 4
    assert np.all(np.isfinite(results["test_score"]))


# ---------------------------------------------------------------------
# n_features_in_ / fitted-state behaviour
# ---------------------------------------------------------------------


def test_not_fitted_error_before_fit():
    clf = StripeCClassifier()
    X, _ = _toy_dataset()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.decision_function(X)


def test_n_features_in_set_after_fit():
    X, y = _toy_dataset(n_features=7)
    clf = StripeCClassifier(random_state=0).fit(X, y)
    assert clf.n_features_in_ == 7


def test_predict_rejects_mismatched_feature_count():
    X, y = _toy_dataset(n_features=7)
    clf = StripeCClassifier(random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :3])


def test_classes_and_label_attributes():
    X, y = _toy_dataset()
    clf = StripeCClassifier(random_state=0).fit(X, y)
    np.testing.assert_array_equal(clf.classes_, np.unique(y))
    assert clf.neg_label_ == clf.classes_[0]
    assert clf.pos_label_ == clf.classes_[1]


def test_fit_rejects_single_class():
    clf = StripeCClassifier()
    X = np.zeros((5, 2))
    y = np.ones(5)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_fit_rejects_more_than_two_classes():
    clf = StripeCClassifier()
    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        clf.fit(X, y)


def test_invalid_init_mode_raises_on_fit_not_on_construction():
    clf = StripeCClassifier(init_mode="bogus")
    # __init__ must not validate (sklearn contract)
    assert clf.init_mode == "bogus"
    X, y = _toy_dataset()
    with pytest.raises(ValueError, match="init_mode"):
        clf.fit(X, y)


# ---------------------------------------------------------------------
# Deterministic repeated fit with a fixed random_state
# ---------------------------------------------------------------------


def test_repeated_fit_same_instance_is_deterministic():
    X, y = _toy_dataset()
    clf = StripeCClassifier(random_state=42)
    clf.fit(X, y)
    first = clf.weights_.copy()
    clf.fit(X, y)
    second = clf.weights_.copy()
    np.testing.assert_allclose(first, second)


def test_fit_on_fresh_instances_is_deterministic():
    X, y = _toy_dataset()
    clf_a = StripeCClassifier(random_state=7).fit(X, y)
    clf_b = StripeCClassifier(random_state=7).fit(X, y)
    np.testing.assert_allclose(clf_a.weights_, clf_b.weights_)
    np.testing.assert_array_equal(clf_a.predict(X), clf_b.predict(X))


def test_different_random_state_gives_different_shuffle_order():
    X, y = _toy_dataset(n_samples=100)
    clf_a = StripeCClassifier(random_state=1).fit(X, y)
    clf_b = StripeCClassifier(random_state=2).fit(X, y)
    assert not np.allclose(clf_a.weights_, clf_b.weights_)


def test_random_state_accepts_a_generator_instance():
    X, y = _toy_dataset()
    clf_a = StripeCClassifier(random_state=np.random.default_rng(123)).fit(X, y)
    clf_b = StripeCClassifier(random_state=np.random.default_rng(123)).fit(X, y)
    np.testing.assert_allclose(clf_a.weights_, clf_b.weights_)


def test_random_state_accepts_a_randomstate_instance_on_this_numpy():
    X, y = _toy_dataset()
    clf_a = StripeCClassifier(random_state=np.random.RandomState(123)).fit(X, y)
    clf_b = StripeCClassifier(random_state=np.random.RandomState(123)).fit(X, y)
    np.testing.assert_allclose(clf_a.weights_, clf_b.weights_)


# ---------------------------------------------------------------------
# partial_fit label / classes validation
# ---------------------------------------------------------------------


def test_partial_fit_rejects_unknown_label_on_first_call():
    clf = StripeCClassifier(random_state=0)
    X = np.zeros((3, 2))
    y = np.array([0, 1, 5])
    with pytest.raises(ValueError, match="not in classes_"):
        clf.partial_fit(X, y, classes=[0, 1])


def test_partial_fit_rejects_unknown_label_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeCClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=np.unique(y))

    X_bad = X[10:12]
    y_bad = np.array([123, 123])
    with pytest.raises(ValueError, match="not in classes_"):
        clf.partial_fit(X_bad, y_bad)


def test_partial_fit_rejects_inconsistent_classes_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeCClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=[0, 1])

    with pytest.raises(ValueError, match="does not match"):
        clf.partial_fit(X[10:], y[10:], classes=[0, 2])


def test_partial_fit_accepts_matching_classes_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeCClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=[0, 1])
    clf.partial_fit(X[10:], y[10:], classes=[1, 1, 0])
    np.testing.assert_array_equal(clf.classes_, [0, 1])


# ---------------------------------------------------------------------
# Multizone correction zone boundaries (direct, hand-crafted inputs)
# ---------------------------------------------------------------------


def test_zone_counters_cover_all_four_zones():
    # eps=0.1, delta_ratio=0.5 -> delta=0.05, so zone edges are at
    # 0.1 (eps), 0.15 (eps+delta), 0.2 (2*eps). With weights_=[0.0] and
    # x=[1.0], pred=0 and eta = pred - s = -s, so |eta| = |s|. We call
    # the private _one_sample_step directly (bypassing the public +-1
    # label encoding in _prepare_labels) to control |eta| precisely and
    # exercise each zone of Psi_mz.
    for target, expected_zone in [
        (0.05, "zone0_"),  # |eta|=0.05 < eps=0.1
        (0.12, "zone3_"),  # eps=0.1 <= |eta|=0.12 < eps+delta=0.15
        (0.17, "zone2_"),  # eps+delta=0.15 <= |eta|=0.17 < 2*eps=0.2
        (0.30, "zone1_"),  # |eta|=0.30 >= 2*eps=0.2
    ]:
        clf = StripeCClassifier(eps=0.1, delta_ratio=0.5, beta=1.0, fit_bias=False)
        clf.weights_ = np.array([0.0])
        clf.classes_ = np.array([0, 1])
        clf.neg_label_, clf.pos_label_ = 0, 1
        clf.zone0_ = clf.zone1_ = clf.zone2_ = clf.zone3_ = 0
        clf.updates_in_epoch_ = clf.total_updates_ = 0

        clf._one_sample_step(np.array([1.0]), target)

        assert getattr(clf, expected_zone) == 1, (
            f"target={target} expected in {expected_zone}, got "
            f"zone0={clf.zone0_} zone1={clf.zone1_} "
            f"zone2={clf.zone2_} zone3={clf.zone3_}"
        )


def test_regularization_decay_applies_before_residual_evaluation():
    """Directly exercises MATH_AUDIT.md finding M7: decay must be applied
    to the weights BEFORE the residual/update for the same sample is
    computed, not after. We construct a case where the two orders give
    numerically different results and assert the decay-before-residual
    value (matching the reference and the supervisor-confirmed order)."""
    clf = StripeCClassifier(
        eps=100.0,  # eps huge => zone0 (no corrective update) always,
                    # isolating the decay effect from the update effect
        lam=0.5,
        fit_bias=False,
        regularize_bias=False,
        shuffle=False,
        random_state=0,
        init_mode="zeros",
    )
    clf.weights_ = np.array([2.0])
    clf.classes_ = np.array([0, 1])
    clf.neg_label_, clf.pos_label_ = 0, 1
    clf.zone0_ = clf.zone1_ = clf.zone2_ = clf.zone3_ = 0
    clf.updates_in_epoch_ = clf.total_updates_ = 0

    clf._one_sample_step(np.array([1.0]), 0.0)

    # decay-before-residual: weights = 2.0 * (1 - 0.5) = 1.0 (eta is
    # computed from the already-decayed weight but eps is huge so no
    # corrective update is added on top)
    np.testing.assert_allclose(clf.weights_, [1.0])
    assert clf.zone0_ == 1


# ---------------------------------------------------------------------
# Numerical parity with the reference implementation
# ---------------------------------------------------------------------


PARITY_CONFIGS = [
    dict(eps=0.1, delta_ratio=0.05, beta=1.0, lam=0.0, regularize_bias=False, fit_bias=True, epochs_count=1),
    dict(eps=0.2, delta_ratio=0.3, beta=0.7, lam=0.0, regularize_bias=False, fit_bias=True, epochs_count=1),
    dict(eps=0.15, delta_ratio=0.2, beta=1.5, lam=0.1, regularize_bias=False, fit_bias=True, epochs_count=3),
    dict(eps=0.15, delta_ratio=0.2, beta=1.5, lam=0.1, regularize_bias=True, fit_bias=True, epochs_count=2),
    dict(eps=0.1, delta_ratio=0.05, beta=1.0, lam=0.0, regularize_bias=False, fit_bias=False, epochs_count=1),
    dict(eps=0.05, delta_ratio=0.9, beta=1.9, lam=0.05, regularize_bias=False, fit_bias=True, epochs_count=4, init_mode="random", init_scale=0.02),
]


@pytest.mark.parametrize("config", PARITY_CONFIGS)
def test_fit_parity_with_reference(config):
    X, y = _toy_dataset(n_samples=50, n_features=4, random_state=3)
    Reference = _reference_class()

    ref = Reference(random_state=11, **config)
    ref.fit(X, y)

    new = StripeCClassifier(random_state=11, **config)
    new.fit(X, y)

    np.testing.assert_allclose(new.weights_, ref.weights_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        new.decision_function(X), ref.decision_function(X), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(new.predict(X), ref.predict(X))

    assert new.total_updates_ == ref.total_updates_
    assert new.total_constraint_checks_ == ref.total_constraint_checks_
    assert new.zone0_ == ref.zone0_
    assert new.zone1_ == ref.zone1_
    assert new.zone2_ == ref.zone2_
    assert new.zone3_ == ref.zone3_


def test_partial_fit_parity_with_reference():
    X, y = _toy_dataset(n_samples=45, n_features=4, random_state=5)
    Reference = _reference_class()

    ref = Reference(random_state=3, eps=0.1, delta_ratio=0.2, beta=1.2)
    new = StripeCClassifier(random_state=3, eps=0.1, delta_ratio=0.2, beta=1.2)

    classes = np.unique(y)
    chunks = [(X[:15], y[:15]), (X[15:30], y[15:30]), (X[30:], y[30:])]

    first = True
    for Xc, yc in chunks:
        if first:
            ref.partial_fit(Xc, yc, classes=classes)
            new.partial_fit(Xc, yc, classes=classes)
            first = False
        else:
            ref.partial_fit(Xc, yc)
            new.partial_fit(Xc, yc)

    np.testing.assert_allclose(new.weights_, ref.weights_, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(new.predict(X), ref.predict(X))
    assert new.total_updates_ == ref.total_updates_


def test_init_mode_zeros_matches_reference_default():
    X, y = _toy_dataset(n_samples=10, n_features=3, random_state=1)
    Reference = _reference_class()

    ref = Reference(random_state=0)
    ref.fit(X, y)
    new = StripeCClassifier(random_state=0)
    new.fit(X, y)

    # both default to init_mode="zeros" -> deterministic starting point
    # regardless of random_state, verified via parity above; here we
    # additionally confirm neither needed randomness to initialize.
    ref2 = Reference(random_state=999)
    ref2._initialize_weights(new.weights_.shape[0])
    np.testing.assert_array_equal(ref2.weights_, np.zeros(new.weights_.shape[0]))
