"""Tests for :class:`stripe_learning.StripeLClassifier` (linear L-system).

Covers: get_params/clone compatibility, the applicable subset of
sklearn's generic estimator checks, Pipeline/GridSearchCV/cross_validate
smoke tests, n_features_in_/fitted-state behaviour, deterministic
repeated fit, and numerical parity against the untouched reference
implementation in ``references/linear_reference.py``.
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

from stripe_learning import StripeLClassifier

from _reference_loader import load_linear_reference


def _reference_class():
    return load_linear_reference().Stripe


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
    "mode": "cumulative",
    "delta_m": 1e-10,
    "epochs_count": 1,
    "fit_bias": True,
    "lam": 0.0,
    "regularize_bias": False,
    "shuffle": True,
    "random_state": None,
    "thresh": 0.0,
}


# ---------------------------------------------------------------------
# get_params / set_params / clone compatibility
# ---------------------------------------------------------------------


def test_default_constructor_has_no_required_arguments():
    # required for sklearn.base.clone / check_estimator: Estimator() must work
    StripeLClassifier()


def test_get_params_matches_constructor_defaults():
    assert StripeLClassifier().get_params() == DEFAULT_PARAMS


def test_set_params_updates_attributes_without_side_effects():
    clf = StripeLClassifier()
    returned = clf.set_params(lam=0.1, epochs_count=3, mode="single")
    assert returned is clf
    assert clf.lam == 0.1
    assert clf.epochs_count == 3
    assert clf.mode == "single"
    # untouched params keep their defaults
    assert clf.delta_m == DEFAULT_PARAMS["delta_m"]


def test_clone_reproduces_params_without_carrying_fitted_state():
    X, y = _toy_dataset()
    clf = StripeLClassifier(random_state=0, lam=0.05, mode="single")
    clf.fit(X, y)

    cloned = clone(clf)

    assert cloned.get_params() == clf.get_params()
    assert not hasattr(cloned, "kappa_hk_")
    assert not hasattr(cloned, "classes_")

    # the clone must be independently fittable and reproduce the same result
    cloned.fit(X, y)
    np.testing.assert_allclose(cloned.kappa_hk_, clf.kappa_hk_)


def test_init_does_not_mutate_or_validate_arguments():
    # sklearn contract: __init__ stores constructor args verbatim, with no
    # type coercion or validation (that belongs in fit()).
    clf = StripeLClassifier(lam="not-yet-a-float", epochs_count="also-not-an-int")
    assert clf.lam == "not-yet-a-float"
    assert clf.epochs_count == "also-not-an-int"


# ---------------------------------------------------------------------
# sklearn generic estimator checks (applicable subset)
# ---------------------------------------------------------------------


@parametrize_with_checks([StripeLClassifier()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


# ---------------------------------------------------------------------
# Pipeline / GridSearchCV / cross_validate smoke tests
# ---------------------------------------------------------------------


def test_pipeline_smoke():
    X, y = _toy_dataset()
    pipe = make_pipeline(StandardScaler(), StripeLClassifier(random_state=0))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)) <= set(np.unique(y))
    scores = pipe.decision_function(X)
    assert scores.shape == y.shape


def test_grid_search_cv_smoke():
    X, y = _toy_dataset(n_samples=80)
    param_grid = {"lam": [0.0, 0.1], "epochs_count": [1, 2]}
    search = GridSearchCV(
        StripeLClassifier(random_state=0),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert search.best_params_["lam"] in param_grid["lam"]
    assert search.best_params_["epochs_count"] in param_grid["epochs_count"]


def test_cross_validate_smoke():
    X, y = _toy_dataset(n_samples=80)
    results = cross_validate(
        StripeLClassifier(random_state=0),
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
    clf = StripeLClassifier()
    X, _ = _toy_dataset()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.decision_function(X)
    with pytest.raises(NotFittedError):
        clf.get_epoch_counters()


def test_n_features_in_set_after_fit():
    X, y = _toy_dataset(n_features=7)
    clf = StripeLClassifier(random_state=0).fit(X, y)
    assert clf.n_features_in_ == 7


def test_predict_rejects_mismatched_feature_count():
    X, y = _toy_dataset(n_features=7)
    clf = StripeLClassifier(random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :3])


def test_classes_and_label_attributes():
    X, y = _toy_dataset()
    clf = StripeLClassifier(random_state=0).fit(X, y)
    np.testing.assert_array_equal(clf.classes_, np.unique(y))
    assert clf.neg_label_ == clf.classes_[0]
    assert clf.pos_label_ == clf.classes_[1]


def test_fit_rejects_single_class():
    clf = StripeLClassifier()
    X = np.zeros((5, 2))
    y = np.ones(5)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_fit_rejects_more_than_two_classes():
    clf = StripeLClassifier()
    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        clf.fit(X, y)


# ---------------------------------------------------------------------
# Deterministic repeated fit with a fixed random_state
# ---------------------------------------------------------------------


def test_repeated_fit_same_instance_is_deterministic():
    X, y = _toy_dataset()
    clf = StripeLClassifier(random_state=42, mode="single")
    clf.fit(X, y)
    first = clf.kappa_hk_.copy()
    clf.fit(X, y)
    second = clf.kappa_hk_.copy()
    np.testing.assert_allclose(first, second)


def test_fit_on_fresh_instances_is_deterministic():
    X, y = _toy_dataset()
    clf_a = StripeLClassifier(random_state=7, mode="cumulative").fit(X, y)
    clf_b = StripeLClassifier(random_state=7, mode="cumulative").fit(X, y)
    np.testing.assert_allclose(clf_a.kappa_hk_, clf_b.kappa_hk_)
    np.testing.assert_array_equal(clf_a.predict(X), clf_b.predict(X))


def test_different_random_state_gives_different_shuffle_order():
    X, y = _toy_dataset(n_samples=100)
    clf_a = StripeLClassifier(random_state=1).fit(X, y)
    clf_b = StripeLClassifier(random_state=2).fit(X, y)
    assert not np.allclose(clf_a.kappa_hk_, clf_b.kappa_hk_)


def test_random_state_accepts_a_generator_instance():
    # numpy.random.default_rng passes a Generator through unchanged; this
    # is guaranteed by numpy.random.default_rng's contract on every
    # supported NumPy version (unlike RandomState coercion, which is
    # version-dependent -- see the random_state docstring).
    X, y = _toy_dataset()
    clf_a = StripeLClassifier(random_state=np.random.default_rng(123)).fit(X, y)
    clf_b = StripeLClassifier(random_state=np.random.default_rng(123)).fit(X, y)
    np.testing.assert_allclose(clf_a.kappa_hk_, clf_b.kappa_hk_)


def test_random_state_accepts_a_randomstate_instance_on_this_numpy():
    # Documents/exercises the RandomState-coercion path described in the
    # random_state docstring, on whichever NumPy is actually installed.
    X, y = _toy_dataset()
    clf_a = StripeLClassifier(random_state=np.random.RandomState(123)).fit(X, y)
    clf_b = StripeLClassifier(random_state=np.random.RandomState(123)).fit(X, y)
    np.testing.assert_allclose(clf_a.kappa_hk_, clf_b.kappa_hk_)


# ---------------------------------------------------------------------
# partial_fit label / classes validation
# ---------------------------------------------------------------------


def test_partial_fit_rejects_unknown_label_on_first_call():
    clf = StripeLClassifier(random_state=0)
    X = np.zeros((3, 2))
    y = np.array([0, 1, 5])  # 5 is not in the declared classes
    with pytest.raises(ValueError, match="not in classes_"):
        clf.partial_fit(X, y, classes=[0, 1])


def test_partial_fit_rejects_unknown_label_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeLClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=np.unique(y))

    X_bad = X[10:12]
    y_bad = np.array([123, 123])  # not one of the fitted classes_
    with pytest.raises(ValueError, match="not in classes_"):
        clf.partial_fit(X_bad, y_bad)


def test_partial_fit_rejects_inconsistent_classes_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeLClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=[0, 1])

    with pytest.raises(ValueError, match="does not match"):
        clf.partial_fit(X[10:], y[10:], classes=[0, 2])


def test_partial_fit_accepts_matching_classes_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeLClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=[0, 1])
    # same classes, re-supplied and even out of order/with a duplicate --
    # should be accepted since np.unique() normalizes it before comparing
    clf.partial_fit(X[10:], y[10:], classes=[1, 1, 0])
    np.testing.assert_array_equal(clf.classes_, [0, 1])


# ---------------------------------------------------------------------
# Numerical parity with the reference implementation
# ---------------------------------------------------------------------


PARITY_CONFIGS = [
    dict(mode="cumulative", lam=0.0, regularize_bias=False, fit_bias=True, epochs_count=1),
    dict(mode="single", lam=0.0, regularize_bias=False, fit_bias=True, epochs_count=1),
    dict(mode="cumulative", lam=0.1, regularize_bias=False, fit_bias=True, epochs_count=3),
    dict(mode="cumulative", lam=0.1, regularize_bias=True, fit_bias=True, epochs_count=2),
    dict(mode="cumulative", lam=0.0, regularize_bias=False, fit_bias=False, epochs_count=1),
    dict(mode="epoch_wise_unconfirmed_string", lam=0.05, regularize_bias=False, fit_bias=True, epochs_count=4),
]


@pytest.mark.parametrize("config", PARITY_CONFIGS)
def test_fit_parity_with_reference(config):
    X, y = _toy_dataset(n_samples=50, n_features=4, random_state=3)
    Reference = _reference_class()

    ref = Reference(random_state=11, **config)
    ref.fit(X, y)

    new = StripeLClassifier(random_state=11, **config)
    new.fit(X, y)

    np.testing.assert_allclose(new.kappa_hk_, ref.kappa_hk, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        new.decision_function(X), ref.decision_function(X), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(new.predict(X), ref.predict(X))

    assert new.total_updates_ == ref.total_updates_
    assert new.total_constraint_checks_ == ref.total_constraint_checks_
    assert new.total_active_samples_ == ref.total_active_samples_
    np.testing.assert_allclose(new.total_update_norm_sum_, ref.total_update_norm_sum_)
    np.testing.assert_allclose(new.total_update_norm_max_, ref.total_update_norm_max_)


def test_partial_fit_parity_with_reference():
    X, y = _toy_dataset(n_samples=45, n_features=4, random_state=5)
    Reference = _reference_class()

    ref = Reference(random_state=3, mode="cumulative")
    new = StripeLClassifier(random_state=3, mode="cumulative")

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

    np.testing.assert_allclose(new.kappa_hk_, ref.kappa_hk, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(new.predict(X), ref.predict(X))
    assert new.total_updates_ == ref.total_updates_


def test_running_average_statistics_parity_with_reference():
    """Directly check the a_hkm/a_hm running-average recursion (paper eq. 10-11)."""
    X, y = _toy_dataset(n_samples=30, n_features=3, random_state=9)
    Reference = _reference_class()

    ref = Reference(random_state=1, mode="cumulative", lam=0.2)
    ref.fit(X, y)

    new = StripeLClassifier(random_state=1, mode="cumulative", lam=0.2)
    new.fit(X, y)

    np.testing.assert_allclose(new.a_hkm_, ref.a_hkm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new.a_hm_, ref.a_hm, rtol=1e-12, atol=1e-12)


def test_regularization_excludes_bias_by_default_matching_reference():
    """Sanity check that the lam/regularize_bias interaction (bias diag
    zeroed by default) is preserved exactly, not just for regularize_bias
    left at its default -- covered again explicitly with lam>0."""
    X, y = _toy_dataset(n_samples=25, n_features=3, random_state=2)
    Reference = _reference_class()

    ref = Reference(random_state=4, mode="cumulative", lam=0.3, regularize_bias=False)
    ref.fit(X, y)
    new = StripeLClassifier(random_state=4, mode="cumulative", lam=0.3, regularize_bias=False)
    new.fit(X, y)

    # bias column/row of A^(m) must NOT include the lam contribution
    assert ref.a_hkm[-1, -1] == pytest.approx(new.a_hkm_[-1, -1])
    np.testing.assert_allclose(new.a_hkm_, ref.a_hkm, rtol=1e-12, atol=1e-12)
