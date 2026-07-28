"""Tests for :class:`stripe_learning.StripeLKernelClassifier` (kernel L-system).

Covers: get_params/clone compatibility, the applicable subset of
sklearn's generic estimator checks, Pipeline/GridSearchCV/cross_validate
smoke tests, n_features_in_/fitted-state behaviour, deterministic
repeated fit, partial_fit label/classes validation, kernel-specific
behaviour (kernel/gamma resolution, dictionary growth), the exactness
of the incremental Gram-matrix accumulation, and numerical parity
against the untouched reference implementation in
``references/kernel_reference.py``.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from stripe_learning import StripeLKernelClassifier

from _reference_loader import load_kernel_reference


def _reference_class():
    return load_kernel_reference().StripeKernel


def _toy_dataset(n_samples=40, n_features=4, random_state=0):
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
    "delta_m": 1e-10,
    "lam": 0.01,
    "kernel": "rbf",
    "gamma": "scale",
    "epochs_count": 1,
    "thresh": 0.0,
    "random_state": None,
}


# ---------------------------------------------------------------------
# get_params / set_params / clone compatibility
# ---------------------------------------------------------------------


def test_default_constructor_has_no_required_arguments():
    StripeLKernelClassifier()


def test_get_params_matches_constructor_defaults():
    assert StripeLKernelClassifier().get_params() == DEFAULT_PARAMS


def test_set_params_updates_attributes_without_side_effects():
    clf = StripeLKernelClassifier()
    returned = clf.set_params(lam=0.2, kernel="linear", epochs_count=3)
    assert returned is clf
    assert clf.lam == 0.2
    assert clf.kernel == "linear"
    assert clf.epochs_count == 3
    assert clf.gamma == DEFAULT_PARAMS["gamma"]


def test_clone_reproduces_params_without_carrying_fitted_state():
    X, y = _toy_dataset()
    clf = StripeLKernelClassifier(random_state=0, lam=0.05)
    clf.fit(X, y)

    cloned = clone(clf)

    assert cloned.get_params() == clf.get_params()
    assert not hasattr(cloned, "alpha_")
    assert not hasattr(cloned, "classes_")

    cloned.fit(X, y)
    np.testing.assert_allclose(cloned.alpha_, clf.alpha_)


def test_init_does_not_mutate_or_validate_arguments():
    clf = StripeLKernelClassifier(lam="not-yet-a-float", delta_m="also-not-a-float")
    assert clf.lam == "not-yet-a-float"
    assert clf.delta_m == "also-not-a-float"


# ---------------------------------------------------------------------
# sklearn generic estimator checks (applicable subset)
# ---------------------------------------------------------------------


@parametrize_with_checks([StripeLKernelClassifier()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


# ---------------------------------------------------------------------
# Pipeline / GridSearchCV / cross_validate smoke tests
# ---------------------------------------------------------------------


def test_pipeline_smoke():
    X, y = _toy_dataset()
    pipe = make_pipeline(StandardScaler(), StripeLKernelClassifier(random_state=0))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)) <= set(np.unique(y))
    scores = pipe.decision_function(X)
    assert scores.shape == y.shape


def test_grid_search_cv_smoke():
    X, y = _toy_dataset(n_samples=60)
    param_grid = {"lam": [0.01, 0.1], "kernel": ["rbf", "linear"]}
    search = GridSearchCV(
        StripeLKernelClassifier(random_state=0),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert search.best_params_["lam"] in param_grid["lam"]
    assert search.best_params_["kernel"] in param_grid["kernel"]


def test_cross_validate_smoke():
    X, y = _toy_dataset(n_samples=60)
    results = cross_validate(
        StripeLKernelClassifier(random_state=0),
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
    clf = StripeLKernelClassifier()
    X, _ = _toy_dataset()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.decision_function(X)


def test_n_features_in_set_after_fit():
    X, y = _toy_dataset(n_features=6)
    clf = StripeLKernelClassifier(random_state=0).fit(X, y)
    assert clf.n_features_in_ == 6


def test_predict_rejects_mismatched_feature_count():
    X, y = _toy_dataset(n_features=6)
    clf = StripeLKernelClassifier(random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :3])


def test_classes_and_label_attributes():
    X, y = _toy_dataset()
    clf = StripeLKernelClassifier(random_state=0).fit(X, y)
    np.testing.assert_array_equal(clf.classes_, np.unique(y))
    assert clf.neg_label_ == clf.classes_[0]
    assert clf.pos_label_ == clf.classes_[1]


def test_fit_rejects_single_class():
    clf = StripeLKernelClassifier()
    X = np.zeros((5, 2))
    y = np.ones(5)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_fit_rejects_more_than_two_classes():
    clf = StripeLKernelClassifier()
    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        clf.fit(X, y)


# ---------------------------------------------------------------------
# Deterministic repeated fit with a fixed random_state
# ---------------------------------------------------------------------


def test_repeated_fit_same_instance_is_deterministic():
    X, y = _toy_dataset()
    clf = StripeLKernelClassifier(random_state=42)
    clf.fit(X, y)
    first = clf.alpha_.copy()
    clf.fit(X, y)
    second = clf.alpha_.copy()
    np.testing.assert_allclose(first, second)


def test_fit_on_fresh_instances_is_deterministic():
    X, y = _toy_dataset()
    clf_a = StripeLKernelClassifier(random_state=7).fit(X, y)
    clf_b = StripeLKernelClassifier(random_state=7).fit(X, y)
    np.testing.assert_allclose(clf_a.alpha_, clf_b.alpha_)
    np.testing.assert_array_equal(clf_a.predict(X), clf_b.predict(X))


def test_random_state_accepts_a_generator_instance():
    X, y = _toy_dataset()
    clf_a = StripeLKernelClassifier(random_state=np.random.default_rng(123)).fit(X, y)
    clf_b = StripeLKernelClassifier(random_state=np.random.default_rng(123)).fit(X, y)
    np.testing.assert_allclose(clf_a.alpha_, clf_b.alpha_)


def test_random_state_accepts_a_randomstate_instance_on_this_numpy():
    X, y = _toy_dataset()
    clf_a = StripeLKernelClassifier(random_state=np.random.RandomState(123)).fit(X, y)
    clf_b = StripeLKernelClassifier(random_state=np.random.RandomState(123)).fit(X, y)
    np.testing.assert_allclose(clf_a.alpha_, clf_b.alpha_)


# ---------------------------------------------------------------------
# partial_fit label / classes validation
# ---------------------------------------------------------------------


def test_partial_fit_rejects_unknown_label_on_first_call():
    clf = StripeLKernelClassifier(random_state=0)
    X = np.zeros((3, 2))
    y = np.array([0, 1, 5])
    with pytest.raises(ValueError, match="not in classes_"):
        clf.partial_fit(X, y, classes=[0, 1])


def test_partial_fit_rejects_unknown_label_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeLKernelClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=np.unique(y))

    X_bad = X[10:12]
    y_bad = np.array([123, 123])
    with pytest.raises(ValueError, match="not in classes_"):
        clf.partial_fit(X_bad, y_bad)


def test_partial_fit_rejects_inconsistent_classes_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeLKernelClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=[0, 1])

    with pytest.raises(ValueError, match="does not match"):
        clf.partial_fit(X[10:], y[10:], classes=[0, 2])


def test_partial_fit_accepts_matching_classes_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeLKernelClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=[0, 1])
    clf.partial_fit(X[10:], y[10:], classes=[1, 1, 0])
    np.testing.assert_array_equal(clf.classes_, [0, 1])


# ---------------------------------------------------------------------
# Kernel-specific behaviour
# ---------------------------------------------------------------------


def test_gamma_scale_matches_sklearn_convention():
    X, y = _toy_dataset(n_samples=25, n_features=3, random_state=1)
    clf = StripeLKernelClassifier(gamma="scale", random_state=0).fit(X, y)
    expected = 1.0 / (X.shape[1] * X.var())
    assert clf.gamma_ == pytest.approx(expected)


def test_gamma_auto_matches_one_over_n_features():
    X, y = _toy_dataset(n_samples=25, n_features=3, random_state=1)
    clf = StripeLKernelClassifier(gamma="auto", random_state=0).fit(X, y)
    assert clf.gamma_ == pytest.approx(1.0 / X.shape[1])


def test_gamma_float_is_used_directly():
    X, y = _toy_dataset(n_samples=25, n_features=3, random_state=1)
    clf = StripeLKernelClassifier(gamma=0.7, random_state=0).fit(X, y)
    assert clf.gamma_ == pytest.approx(0.7)


def test_decision_function_matches_manual_rbf_kernel_expansion():
    X, y = _toy_dataset(n_samples=20, n_features=3, random_state=2)
    clf = StripeLKernelClassifier(kernel="rbf", gamma=0.5, random_state=0).fit(X, y)
    expected = rbf_kernel(X, clf.X_train_, gamma=0.5).dot(clf.alpha_)
    np.testing.assert_allclose(clf.decision_function(X), expected, rtol=1e-12)


def test_decision_function_matches_manual_linear_kernel_expansion():
    X, y = _toy_dataset(n_samples=20, n_features=3, random_state=2)
    clf = StripeLKernelClassifier(kernel="linear", random_state=0).fit(X, y)
    expected = linear_kernel(X, clf.X_train_).dot(clf.alpha_)
    np.testing.assert_allclose(clf.decision_function(X), expected, rtol=1e-12)


def test_dictionary_grows_by_one_per_accumulated_sample():
    X, y = _toy_dataset(n_samples=15, n_features=3, random_state=4)
    clf = StripeLKernelClassifier(random_state=0)
    clf.partial_fit(X[:5], y[:5], classes=np.unique(y))
    assert clf.m_ == 5
    assert clf.X_train_.shape == (5, 3)
    assert clf.alpha_.shape == (5,)

    clf.partial_fit(X[5:9], y[5:9])
    assert clf.m_ == 9
    assert clf.X_train_.shape == (9, 3)
    assert clf.alpha_.shape == (9,)


def test_no_bias_feature_appended_matching_paper_kernel_formulation():
    # unlike the linear estimators, the kernel model has no intercept
    # term (see MATH_AUDIT.md finding F3); X_train_ must retain exactly
    # n_features_in_ columns, never n_features_in_ + 1.
    X, y = _toy_dataset(n_samples=10, n_features=3, random_state=6)
    clf = StripeLKernelClassifier(random_state=0).fit(X, y)
    assert clf.X_train_.shape[1] == clf.n_features_in_ == 3


# ---------------------------------------------------------------------
# Exactness of the incremental Gram-matrix accumulation
# (MATH_AUDIT.md Section 3.3: verified by hand-derivation, checked here
# against a from-scratch recomputation rather than only against the
# reference implementation)
# ---------------------------------------------------------------------


def test_incremental_accumulation_matches_recompute_from_scratch():
    X, y = _toy_dataset(n_samples=12, n_features=3, random_state=8)
    clf = StripeLKernelClassifier(kernel="rbf", gamma=0.3, random_state=0)
    clf.partial_fit(X, y, classes=np.unique(y))

    K_full = rbf_kernel(clf.X_train_, clf.X_train_, gamma=0.3)
    np.testing.assert_allclose(clf.K_all_, K_full, rtol=1e-10, atol=1e-12)

    B_full = K_full @ K_full
    np.testing.assert_allclose(clf.B_unnormalized_, B_full, rtol=1e-8, atol=1e-10)

    c_full = K_full @ clf.y_train_internal_
    np.testing.assert_allclose(clf.c_unnormalized_, c_full, rtol=1e-8, atol=1e-10)


def test_incremental_accumulation_matches_recompute_from_scratch_incrementally():
    """Same check as above, but verified after every single accumulated
    point, not just at the end, to rule out an error that happens to
    cancel out by the final point."""
    X, y = _toy_dataset(n_samples=8, n_features=3, random_state=9)
    clf = StripeLKernelClassifier(kernel="rbf", gamma=0.4, random_state=0)

    for i in range(X.shape[0]):
        if i == 0:
            clf.partial_fit(X[:1], y[:1], classes=np.unique(y))
        else:
            clf.partial_fit(X[i : i + 1], y[i : i + 1])

        K_full = rbf_kernel(clf.X_train_, clf.X_train_, gamma=0.4)
        np.testing.assert_allclose(clf.K_all_, K_full, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(
            clf.B_unnormalized_, K_full @ K_full, rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(
            clf.c_unnormalized_,
            K_full @ clf.y_train_internal_,
            rtol=1e-7,
            atol=1e-9,
        )


def test_regularization_included_in_projection_normalization():
    """MATH_AUDIT.md finding M12: the row-norm used to normalize each
    Stripe projection step must be taken on the *regularized* system
    (A = B/M + lam*I), not the raw unregularized Gram statistics."""
    X, y = _toy_dataset(n_samples=10, n_features=3, random_state=10)
    clf = StripeLKernelClassifier(lam=0.7, random_state=0)
    clf.partial_fit(X, y, classes=np.unique(y))
    clf._build_cached_system_if_needed()

    A_regularized = clf.B_unnormalized_ / clf.m_ + np.eye(clf.m_) * 0.7
    expected_row_norms_sq = np.einsum("ij,ij->i", A_regularized, A_regularized)
    np.testing.assert_allclose(clf.row_norms_sq_, expected_row_norms_sq, rtol=1e-10)


# ---------------------------------------------------------------------
# Numerical parity with the reference implementation
# ---------------------------------------------------------------------


PARITY_CONFIGS = [
    dict(kernel="rbf", gamma="scale", lam=0.01, epochs_count=1),
    dict(kernel="rbf", gamma="auto", lam=0.1, epochs_count=3),
    dict(kernel="rbf", gamma=0.5, lam=0.0, epochs_count=2),
    dict(kernel="linear", gamma="scale", lam=0.05, epochs_count=2),
]


@pytest.mark.parametrize("config", PARITY_CONFIGS)
def test_fit_parity_with_reference(config):
    X, y = _toy_dataset(n_samples=30, n_features=4, random_state=3)
    Reference = _reference_class()

    epochs_count = config.pop("epochs_count")
    ref = Reference(random_state=11, **config)
    ref.fit(X, y, epochs_count=epochs_count)

    new = StripeLKernelClassifier(random_state=11, epochs_count=epochs_count, **config)
    new.fit(X, y)

    np.testing.assert_allclose(new.alpha_, ref.alpha_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        new.decision_function(X), ref.decision_function(X), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(new.predict(X), ref.predict(X))
    assert new.total_updates_ == ref.total_updates_
    assert new.m_ == ref.m_
    np.testing.assert_allclose(new.gamma_, ref.gamma_)


def test_partial_fit_parity_with_reference():
    """new.partial_fit(X, y) combines "accumulate" + "run one epoch" in
    a single call (IMPLEMENTATION_PLAN.md section 5.2); the reference
    only offers these as two separate calls
    (partial_fit(..., accumulate=True) then partial_fit()). We drive
    the reference through the equivalent two-step sequence and check
    the results match exactly, confirming the consolidation is a pure
    API change with no numerical effect."""
    X, y = _toy_dataset(n_samples=24, n_features=3, random_state=5)
    Reference = _reference_class()

    ref = Reference(random_state=3, lam=0.02)
    new = StripeLKernelClassifier(random_state=3, lam=0.02)

    classes = np.unique(y)
    chunks = [(X[:8], y[:8]), (X[8:16], y[8:16]), (X[16:], y[16:])]

    first = True
    for Xc, yc in chunks:
        if first:
            ref.partial_fit(Xc, yc, classes=classes, accumulate=True)
            new.partial_fit(Xc, yc, classes=classes)
            first = False
        else:
            ref.partial_fit(Xc, yc, accumulate=True)
            new.partial_fit(Xc, yc)
        ref.partial_fit()  # run exactly one Stripe epoch, matching new's per-call epoch

    np.testing.assert_allclose(new.alpha_, ref.alpha_, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(new.predict(X), ref.predict(X))
    assert new.total_updates_ == ref.total_updates_


def test_accumulation_parity_with_reference_block_matrix_update():
    """Directly checks that our _accumulate_one port produces the same
    K_all_/B_unnormalized_/c_unnormalized_ as the reference's, matching
    MATH_AUDIT.md Section 3.3's exactness derivation."""
    X, y = _toy_dataset(n_samples=14, n_features=3, random_state=7)
    Reference = _reference_class()

    ref = Reference(random_state=1)
    ref.fit(X, y, epochs_count=0)  # accumulate only, no correction epochs

    new = StripeLKernelClassifier(random_state=1, epochs_count=0)
    new.fit(X, y)

    np.testing.assert_allclose(new.K_all_, ref.K_all_, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        new.B_unnormalized_, ref.B_unnormalized_, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        new.c_unnormalized_, ref.c_unnormalized_, rtol=1e-10, atol=1e-12
    )
    # epochs_count=0 means no correction ever ran, so alpha_ is all-zero
    # on both sides -- this is the exact footgun IMPLEMENTATION_PLAN.md
    # section 5.1 fixed by changing the *default* to 1, not by changing
    # what epochs_count=0 itself does.
    np.testing.assert_array_equal(new.alpha_, np.zeros_like(new.alpha_))
    np.testing.assert_array_equal(ref.alpha_, np.zeros_like(ref.alpha_))
