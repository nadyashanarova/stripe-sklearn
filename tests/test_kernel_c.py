"""Tests for :class:`stripe_learning.StripeCKernelClassifier` (kernel C-system).

Covers: get_params/clone compatibility, the applicable subset of
sklearn's generic estimator checks, Pipeline/GridSearchCV/cross_validate
smoke tests, n_features_in_/fitted-state behaviour, deterministic
repeated fit, partial_fit label/classes validation, kernel-specific
behaviour (kernel/gamma resolution and the kernel-dispatch fallback
direction, which is the opposite of StripeLKernelClassifier's), the
exactness of the incremental Gram-matrix/row-norm accumulation, the
multizone correction zones, the decay-before-residual regularization
order, and numerical parity against the untouched reference
implementation in ``references/kernel_reference.py``.
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

from stripe_learning import StripeCKernelClassifier

from _reference_loader import load_kernel_reference


def _reference_class():
    return load_kernel_reference().CSystemKernelEpochWise


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
    "eps": 0.1,
    "delta_ratio": 0.5,
    "beta_param": 1.0,
    "lam": 0.0,
    "delta_m": 1e-12,
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
    StripeCKernelClassifier()


def test_get_params_matches_constructor_defaults():
    assert StripeCKernelClassifier().get_params() == DEFAULT_PARAMS


def test_set_params_updates_attributes_without_side_effects():
    clf = StripeCKernelClassifier()
    returned = clf.set_params(lam=0.2, kernel="linear", eps=0.3)
    assert returned is clf
    assert clf.lam == 0.2
    assert clf.kernel == "linear"
    assert clf.eps == 0.3
    assert clf.delta_ratio == DEFAULT_PARAMS["delta_ratio"]


def test_clone_reproduces_params_without_carrying_fitted_state():
    X, y = _toy_dataset()
    clf = StripeCKernelClassifier(random_state=0, lam=0.05)
    clf.fit(X, y)

    cloned = clone(clf)

    assert cloned.get_params() == clf.get_params()
    assert not hasattr(cloned, "alpha_")
    assert not hasattr(cloned, "classes_")

    cloned.fit(X, y)
    np.testing.assert_allclose(cloned.alpha_, clf.alpha_)


def test_init_does_not_mutate_or_validate_arguments():
    clf = StripeCKernelClassifier(lam="not-yet-a-float", eps="also-not-a-float")
    assert clf.lam == "not-yet-a-float"
    assert clf.eps == "also-not-a-float"


def test_delta_property_matches_eps_times_delta_ratio():
    clf = StripeCKernelClassifier(eps=0.2, delta_ratio=0.25)
    assert clf.delta == pytest.approx(0.05)


# ---------------------------------------------------------------------
# sklearn generic estimator checks (applicable subset)
# ---------------------------------------------------------------------


@parametrize_with_checks([StripeCKernelClassifier()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


# ---------------------------------------------------------------------
# Pipeline / GridSearchCV / cross_validate smoke tests
# ---------------------------------------------------------------------


def test_pipeline_smoke():
    X, y = _toy_dataset()
    pipe = make_pipeline(StandardScaler(), StripeCKernelClassifier(random_state=0))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)) <= set(np.unique(y))
    scores = pipe.decision_function(X)
    assert scores.shape == y.shape


def test_grid_search_cv_smoke():
    X, y = _toy_dataset(n_samples=60)
    param_grid = {"eps": [0.05, 0.2], "beta_param": [0.5, 1.5]}
    search = GridSearchCV(
        StripeCKernelClassifier(random_state=0),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
    )
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert search.best_params_["eps"] in param_grid["eps"]
    assert search.best_params_["beta_param"] in param_grid["beta_param"]


def test_cross_validate_smoke():
    X, y = _toy_dataset(n_samples=60)
    results = cross_validate(
        StripeCKernelClassifier(random_state=0),
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
    clf = StripeCKernelClassifier()
    X, _ = _toy_dataset()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.decision_function(X)


def test_n_features_in_set_after_fit():
    X, y = _toy_dataset(n_features=6)
    clf = StripeCKernelClassifier(random_state=0).fit(X, y)
    assert clf.n_features_in_ == 6


def test_predict_rejects_mismatched_feature_count():
    X, y = _toy_dataset(n_features=6)
    clf = StripeCKernelClassifier(random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :3])


def test_classes_and_label_attributes():
    X, y = _toy_dataset()
    clf = StripeCKernelClassifier(random_state=0).fit(X, y)
    np.testing.assert_array_equal(clf.classes_, np.unique(y))
    assert clf.neg_label_ == clf.classes_[0]
    assert clf.pos_label_ == clf.classes_[1]


def test_fit_rejects_single_class():
    clf = StripeCKernelClassifier()
    X = np.zeros((5, 2))
    y = np.ones(5)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_fit_rejects_more_than_two_classes():
    clf = StripeCKernelClassifier()
    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        clf.fit(X, y)


# ---------------------------------------------------------------------
# Deterministic repeated fit with a fixed random_state
# ---------------------------------------------------------------------


def test_repeated_fit_same_instance_is_deterministic():
    X, y = _toy_dataset()
    clf = StripeCKernelClassifier(random_state=42)
    clf.fit(X, y)
    first = clf.alpha_.copy()
    clf.fit(X, y)
    second = clf.alpha_.copy()
    np.testing.assert_allclose(first, second)


def test_fit_on_fresh_instances_is_deterministic():
    X, y = _toy_dataset()
    clf_a = StripeCKernelClassifier(random_state=7).fit(X, y)
    clf_b = StripeCKernelClassifier(random_state=7).fit(X, y)
    np.testing.assert_allclose(clf_a.alpha_, clf_b.alpha_)
    np.testing.assert_array_equal(clf_a.predict(X), clf_b.predict(X))


def test_random_state_accepts_a_generator_instance():
    X, y = _toy_dataset()
    clf_a = StripeCKernelClassifier(random_state=np.random.default_rng(123)).fit(X, y)
    clf_b = StripeCKernelClassifier(random_state=np.random.default_rng(123)).fit(X, y)
    np.testing.assert_allclose(clf_a.alpha_, clf_b.alpha_)


def test_random_state_accepts_a_randomstate_instance_on_this_numpy():
    X, y = _toy_dataset()
    clf_a = StripeCKernelClassifier(random_state=np.random.RandomState(123)).fit(X, y)
    clf_b = StripeCKernelClassifier(random_state=np.random.RandomState(123)).fit(X, y)
    np.testing.assert_allclose(clf_a.alpha_, clf_b.alpha_)


# ---------------------------------------------------------------------
# partial_fit label / classes validation
# ---------------------------------------------------------------------


def test_partial_fit_rejects_unknown_label_on_first_call():
    clf = StripeCKernelClassifier(random_state=0)
    X = np.zeros((3, 2))
    y = np.array([0, 1, 5])
    with pytest.raises(ValueError, match="not in classes_"):
        clf.partial_fit(X, y, classes=[0, 1])


def test_partial_fit_rejects_unknown_label_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeCKernelClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=np.unique(y))

    X_bad = X[10:12]
    y_bad = np.array([123, 123])
    with pytest.raises(ValueError, match="not in classes_"):
        clf.partial_fit(X_bad, y_bad)


def test_partial_fit_rejects_inconsistent_classes_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeCKernelClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=[0, 1])

    with pytest.raises(ValueError, match="does not match"):
        clf.partial_fit(X[10:], y[10:], classes=[0, 2])


def test_partial_fit_accepts_matching_classes_on_later_call():
    X, y = _toy_dataset(n_samples=20)
    clf = StripeCKernelClassifier(random_state=0)
    clf.partial_fit(X[:10], y[:10], classes=[0, 1])
    clf.partial_fit(X[10:], y[10:], classes=[1, 1, 0])
    np.testing.assert_array_equal(clf.classes_, [0, 1])


# ---------------------------------------------------------------------
# Kernel-specific behaviour
# ---------------------------------------------------------------------


def test_gamma_scale_matches_sklearn_convention():
    X, y = _toy_dataset(n_samples=25, n_features=3, random_state=1)
    clf = StripeCKernelClassifier(gamma="scale", random_state=0).fit(X, y)
    expected = 1.0 / (X.shape[1] * X.var())
    assert clf.gamma_ == pytest.approx(expected)


def test_gamma_auto_matches_one_over_n_features():
    X, y = _toy_dataset(n_samples=25, n_features=3, random_state=1)
    clf = StripeCKernelClassifier(gamma="auto", random_state=0).fit(X, y)
    assert clf.gamma_ == pytest.approx(1.0 / X.shape[1])


def test_gamma_float_is_used_directly():
    X, y = _toy_dataset(n_samples=25, n_features=3, random_state=1)
    clf = StripeCKernelClassifier(gamma=0.7, random_state=0).fit(X, y)
    assert clf.gamma_ == pytest.approx(0.7)


def test_unrecognized_kernel_string_falls_back_to_linear():
    """Unlike StripeLKernelClassifier (falls back to "rbf"), this class's
    _get_kernel_func falls back to "linear" for any kernel value other
    than the literal string "rbf" -- matching the reference exactly
    (see the class docstring)."""
    X, y = _toy_dataset(n_samples=15, n_features=3, random_state=2)
    clf_bogus = StripeCKernelClassifier(kernel="bogus", random_state=0).fit(X, y)
    clf_linear = StripeCKernelClassifier(kernel="linear", random_state=0).fit(X, y)
    np.testing.assert_allclose(clf_bogus.alpha_, clf_linear.alpha_)
    np.testing.assert_allclose(clf_bogus.K_all_, clf_linear.K_all_)


def test_decision_function_matches_manual_rbf_kernel_expansion():
    X, y = _toy_dataset(n_samples=20, n_features=3, random_state=2)
    clf = StripeCKernelClassifier(kernel="rbf", gamma=0.5, random_state=0).fit(X, y)
    expected = rbf_kernel(X, clf.X_train_, gamma=0.5).dot(clf.alpha_)
    np.testing.assert_allclose(clf.decision_function(X), expected, rtol=1e-12)


def test_decision_function_matches_manual_linear_kernel_expansion():
    X, y = _toy_dataset(n_samples=20, n_features=3, random_state=2)
    clf = StripeCKernelClassifier(kernel="linear", random_state=0).fit(X, y)
    expected = linear_kernel(X, clf.X_train_).dot(clf.alpha_)
    np.testing.assert_allclose(clf.decision_function(X), expected, rtol=1e-12)


def test_dictionary_grows_by_one_per_accumulated_sample():
    X, y = _toy_dataset(n_samples=15, n_features=3, random_state=4)
    clf = StripeCKernelClassifier(random_state=0)
    clf.partial_fit(X[:5], y[:5], classes=np.unique(y))
    assert clf.m_ == 5
    assert clf.X_train_.shape == (5, 3)
    assert clf.alpha_.shape == (5,)

    clf.partial_fit(X[5:9], y[5:9])
    assert clf.m_ == 9
    assert clf.X_train_.shape == (9, 3)
    assert clf.alpha_.shape == (9,)


def test_no_bias_feature_appended_matching_paper_kernel_formulation():
    X, y = _toy_dataset(n_samples=10, n_features=3, random_state=6)
    clf = StripeCKernelClassifier(random_state=0).fit(X, y)
    assert clf.X_train_.shape[1] == clf.n_features_in_ == 3


def test_no_per_zone_counters_matching_reference():
    """Unlike StripeCClassifier, the kernel C-system reference tracks no
    zone0_..zone3_ counters -- only updates_in_epoch_/total_updates_.
    This asymmetry is preserved exactly (IMPLEMENTATION_PLAN.md section
    5.12); assert none of the linear C-system's zone attributes exist
    here."""
    X, y = _toy_dataset(n_samples=10, random_state=0)
    clf = StripeCKernelClassifier(random_state=0).fit(X, y)
    for attr in ("zone0_", "zone1_", "zone2_", "zone3_"):
        assert not hasattr(clf, attr)


# ---------------------------------------------------------------------
# Multizone correction zones (direct, hand-crafted state)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,expect_update,expected_alpha",
    [
        (0.05, False, 0.0),  # |eta|=0.05 < eps=0.1 -> zone0, no update
        (0.12, True, 0.02),  # eps=0.1 <= |eta|=0.12 < eps+delta=0.15 -> buffer zone
        (0.17, True, 0.14),  # eps+delta=0.15 <= |eta|=0.17 < 2*eps=0.2 -> reflection
        (0.30, True, 0.30),  # |eta|=0.30 >= 2*eps=0.2 -> full projection
    ],
)
def test_zone_updates_match_hand_computed_values(target, expect_update, expected_alpha):
    # eps=0.1, delta_ratio=0.5 -> delta=0.05; a single dictionary point
    # with K_all_=[[1.0]] (norm_sq=1) and alpha_=[0.0] gives
    # eta = y_train_signed_[0] - alpha.K_all_[0] = target exactly.
    clf = StripeCKernelClassifier(eps=0.1, delta_ratio=0.5, beta_param=1.0, random_state=0)
    clf.alpha_ = np.array([0.0])
    clf.K_all_ = np.array([[1.0]])
    clf.norms_sq_C_ = np.array([1.0])
    clf.y_train_signed_ = np.array([target])
    clf.m_ = 1
    clf.updates_in_epoch_ = clf.total_updates_ = 0
    clf._rng = np.random.default_rng(0)

    clf._run_one_epoch()

    np.testing.assert_allclose(clf.alpha_, [expected_alpha], atol=1e-12)
    assert (clf.total_updates_ == 1) is expect_update


def test_regularization_decay_applies_before_residual_evaluation():
    """MATH_AUDIT.md finding on kernel-C's decay order: decay must be
    applied to alpha_ BEFORE eta is computed for the same constraint,
    every iteration -- and this already matched the reference and the
    paper's Appendix B.4 text without needing a supervisor
    clarification (unlike the linear C-system)."""
    clf = StripeCKernelClassifier(
        eps=100.0,  # huge eps -> always zone0 after decay, isolating the
        # decay effect from the correction effect
        lam=0.5,
        random_state=0,
    )
    clf.alpha_ = np.array([2.0])
    clf.K_all_ = np.array([[1.0]])
    clf.norms_sq_C_ = np.array([1.0])
    clf.y_train_signed_ = np.array([0.0])
    clf.m_ = 1
    clf.updates_in_epoch_ = clf.total_updates_ = 0
    clf._rng = np.random.default_rng(0)

    clf._run_one_epoch()

    # decay-before-residual: alpha = 2.0 * (1 - 0.5) = 1.0, eps=100 means
    # no corrective term is added on top
    np.testing.assert_allclose(clf.alpha_, [1.0])
    assert clf.total_updates_ == 0


# ---------------------------------------------------------------------
# Exactness of the incremental Gram-matrix / row-norm accumulation
# ---------------------------------------------------------------------


def test_incremental_accumulation_matches_recompute_from_scratch():
    X, y = _toy_dataset(n_samples=12, n_features=3, random_state=8)
    clf = StripeCKernelClassifier(kernel="rbf", gamma=0.3, random_state=0)
    clf.partial_fit(X, y, classes=np.unique(y))

    K_full = rbf_kernel(clf.X_train_, clf.X_train_, gamma=0.3)
    np.testing.assert_allclose(clf.K_all_, K_full, rtol=1e-10, atol=1e-12)

    expected_norms_sq = (K_full**2).sum(axis=1)
    np.testing.assert_allclose(clf.norms_sq_C_, expected_norms_sq, rtol=1e-8, atol=1e-10)


def test_incremental_accumulation_matches_recompute_from_scratch_incrementally():
    X, y = _toy_dataset(n_samples=8, n_features=3, random_state=9)
    clf = StripeCKernelClassifier(kernel="rbf", gamma=0.4, random_state=0)

    for i in range(X.shape[0]):
        if i == 0:
            clf.partial_fit(X[:1], y[:1], classes=np.unique(y))
        else:
            clf.partial_fit(X[i : i + 1], y[i : i + 1])

        K_full = rbf_kernel(clf.X_train_, clf.X_train_, gamma=0.4)
        np.testing.assert_allclose(clf.K_all_, K_full, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(
            clf.norms_sq_C_, (K_full**2).sum(axis=1), rtol=1e-7, atol=1e-9
        )


# ---------------------------------------------------------------------
# Numerical parity with the reference implementation
# ---------------------------------------------------------------------


PARITY_CONFIGS = [
    dict(kernel="rbf", gamma="scale", eps=0.1, delta_ratio=0.5, beta_param=1.0, lam=0.0, epochs_count=1),
    dict(kernel="rbf", gamma="auto", eps=0.2, delta_ratio=0.3, beta_param=0.7, lam=0.1, epochs_count=3),
    dict(kernel="rbf", gamma=0.5, eps=0.15, delta_ratio=0.2, beta_param=1.5, lam=0.05, epochs_count=2),
    dict(kernel="linear", gamma="scale", eps=0.05, delta_ratio=0.9, beta_param=1.9, lam=0.02, epochs_count=2),
]


@pytest.mark.parametrize("config", PARITY_CONFIGS)
def test_fit_parity_with_reference(config):
    X, y = _toy_dataset(n_samples=30, n_features=4, random_state=3)
    Reference = _reference_class()

    epochs_count = config.pop("epochs_count")
    ref = Reference(random_state=11, **config)
    ref.fit(X, y, epochs_count=epochs_count)

    new = StripeCKernelClassifier(random_state=11, epochs_count=epochs_count, **config)
    new.fit(X, y)

    np.testing.assert_allclose(new.alpha_, ref.alpha_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        new.decision_function(X), ref.decision_function(X), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(new.predict(X), ref.predict(X))
    assert new.total_updates_ == ref.total_updates_
    assert new.m_ == ref.m_
    np.testing.assert_allclose(new.gamma_, ref.gamma_)
    assert new.delta == pytest.approx(ref.delta_)


def test_partial_fit_parity_with_reference():
    """Same rationale as StripeLKernelClassifier's equivalent test: the
    reference only offers accumulate-then-epoch as two separate calls;
    we drive it through that exact two-step sequence and check it
    matches our single consolidated partial_fit(X, y) call."""
    X, y = _toy_dataset(n_samples=24, n_features=3, random_state=5)
    Reference = _reference_class()

    ref = Reference(random_state=3, lam=0.02, eps=0.1, delta_ratio=0.3)
    new = StripeCKernelClassifier(random_state=3, lam=0.02, eps=0.1, delta_ratio=0.3)

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
        ref.partial_fit()  # run exactly one Stripe epoch

    np.testing.assert_allclose(new.alpha_, ref.alpha_, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(new.predict(X), ref.predict(X))
    assert new.total_updates_ == ref.total_updates_


def test_accumulation_parity_with_reference_block_matrix_update():
    X, y = _toy_dataset(n_samples=14, n_features=3, random_state=7)
    Reference = _reference_class()

    ref = Reference(random_state=1)
    ref.fit(X, y, epochs_count=0)  # accumulate only

    new = StripeCKernelClassifier(random_state=1, epochs_count=0)
    new.fit(X, y)

    np.testing.assert_allclose(new.K_all_, ref.K_all_, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        new.norms_sq_C_, ref.norms_sq_C_, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(new.alpha_, np.zeros_like(new.alpha_))
    np.testing.assert_array_equal(ref.alpha_, np.zeros_like(ref.alpha_))
