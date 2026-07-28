"""Linear L-system Stripe classifier (Yakubovich).

Ported from ``references/linear_reference.py`` (class ``Stripe``). See
``IMPLEMENTATION_PLAN.md`` and ``MATH_AUDIT.md`` at the repository root
for the full analysis this port is based on. Every approved deviation
from the reference implementation is an sklearn/API-compliance change
only (constructor shape, attribute naming, input validation, dead-code
removal); the projection mathematics -- the running-average recursion
for ``a_hkm``/``a_hm``, the coordinate-wise residual, the basic-Stripe
update, and the ``delta_m`` trigger -- is preserved exactly.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class StripeLClassifier(ClassifierMixin, BaseEstimator):
    """Yakubovich Stripe / L-system linear binary classifier.

    Learning is posed as a system of hyperslab constraints derived from
    the relaxed stationarity condition of a regularized quadratic loss
    (paper eq. 2, 10-11). Parameters are corrected only when a
    coordinate-wise residual violates its tolerance, via the basic
    Stripe projection step (paper eq. 6), with ``delta_m`` playing the
    role of the tolerance ``epsilon`` in that equation (see
    ``MATH_AUDIT.md`` finding F1/M10: the improved Stripe variant,
    paper eq. 7, is intentionally not implemented, so Theorem 1's
    finite-convergence guarantee does not directly cover this
    estimator).

    Parameters
    ----------
    mode : str, default="cumulative"
        Controls how the running statistics ``a_hkm``/``a_hm`` are
        reset between samples/epochs. ``"single"`` rebuilds the
        statistics from the current sample only; ``"cumulative"``
        accumulates over all samples ever seen (including across
        ``partial_fit`` calls); any other value falls back to a
        per-epoch reset, matching the reference implementation exactly
        (this string is intentionally left unvalidated -- see
        ``IMPLEMENTATION_PLAN.md`` section 5.8).
    delta_m : float, default=1e-10
        Update-trigger tolerance: a coordinate is corrected whenever
        the magnitude of its residual is at least ``delta_m``. Plays
        the role of ``epsilon`` in the paper's basic Stripe update
        (eq. 6); see the class docstring and ``MATH_AUDIT.md``
        finding F1/M10.
    epochs_count : int, default=1
        Number of passes over the training data performed by ``fit``.
    fit_bias : bool, default=True
        Whether to append a constant feature (bias/intercept) to the
        input.
    lam : float, default=0.0
        L2 regularization strength added to the running second-moment
        matrix (paper eq. 10).
    regularize_bias : bool, default=False
        Whether the bias coordinate (when ``fit_bias=True``) is
        included in the L2 regularization.
    shuffle : bool, default=True
        Whether to shuffle sample order (and, within each sample,
        coordinate order) using the estimator's random state.
    random_state : int, array-like of ints, numpy.random.SeedSequence, \
            numpy.random.BitGenerator, numpy.random.Generator, or None, \
            default=None
        Controls the shuffling randomness. Passed straight through to
        ``numpy.random.default_rng`` to construct the estimator's random
        generator, so the exact set of accepted seed types is whatever
        the installed NumPy version's ``default_rng`` accepts for its
        ``seed`` argument -- this always includes ``int`` and ``None``.
        A legacy ``numpy.random.RandomState`` instance is also accepted
        on NumPy versions where ``default_rng`` documents that
        coercion; consult the installed NumPy's own documentation if in
        doubt. Note this estimator does **not** use scikit-learn's
        ``sklearn.utils.check_random_state``, so it does not follow the
        broader RandomState-coercion convention used by most other
        scikit-learn estimators. The random generator is reseeded at
        the start of every ``fit`` call (and on the first
        ``partial_fit`` call), so repeated ``fit`` calls with a fixed
        ``random_state`` are reproducible.
    thresh : float, default=0.0
        Decision threshold ``tau`` applied to ``decision_function`` in
        ``predict`` (paper eq. 1).

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The two class labels seen during ``fit``, sorted ascending;
        ``classes_[0]`` is the negative class, ``classes_[1]`` the
        positive class.
    n_features_in_ : int
        Number of features seen during ``fit`` (excluding the bias
        feature, if any).
    kappa_hk_ : ndarray of shape (n_features_in_ + int(fit_bias),)
        Learned parameter vector.
    a_hkm_ : ndarray of shape (n_features_in_ + int(fit_bias), \
            n_features_in_ + int(fit_bias))
        Running estimate of the regularized second-moment matrix
        ``A^(m)`` (paper eq. 10).
    a_hm_ : ndarray of shape (n_features_in_ + int(fit_bias),)
        Running estimate of the label-correlation vector ``B^(m)``
        (paper eq. 11).
    m_ : int
        Internal running-average sample counter (``-1`` before any
        sample has been processed).
    updates_in_epoch_, total_updates_ : int
        Number of coordinate corrections in the most recent
        epoch/call, and cumulatively since the last ``fit``/
        ``partial_fit``-triggered (re)initialization.
    constraint_checks_in_epoch_, total_constraint_checks_ : int
        Number of coordinate constraint evaluations.
    active_samples_in_epoch_, total_active_samples_ : int
        Number of samples that triggered at least one correction.
    update_norm_sum_in_epoch_, update_norm_max_in_epoch_,
    total_update_norm_sum_, total_update_norm_max_ : float
        Sum/max of the L2 norm of each individual coordinate
        correction.
    """

    def __init__(
        self,
        mode="cumulative",
        delta_m=1e-10,
        epochs_count=1,
        fit_bias=True,
        lam=0.0,
        regularize_bias=False,
        shuffle=True,
        random_state=None,
        thresh=0.0,
    ):
        self.mode = mode
        self.delta_m = delta_m
        self.epochs_count = epochs_count
        self.fit_bias = fit_bias
        self.lam = lam
        self.regularize_bias = regularize_bias
        self.shuffle = shuffle
        self.random_state = random_state
        self.thresh = thresh

    # ------------------------------------------------------------------
    # Counter helpers (verbatim from the reference implementation)
    # ------------------------------------------------------------------

    def _reset_epoch_counters(self):
        self.updates_in_epoch_ = 0
        self.constraint_checks_in_epoch_ = 0
        self.active_samples_in_epoch_ = 0
        self.update_norm_sum_in_epoch_ = 0.0
        self.update_norm_max_in_epoch_ = 0.0

    def _reset_total_counters(self):
        self.total_updates_ = 0
        self.total_constraint_checks_ = 0
        self.total_active_samples_ = 0
        self.total_update_norm_sum_ = 0.0
        self.total_update_norm_max_ = 0.0

    @property
    def update_rate_in_epoch_(self):
        if self.constraint_checks_in_epoch_ == 0:
            return np.nan
        return self.updates_in_epoch_ / self.constraint_checks_in_epoch_

    @property
    def active_sample_rate_in_epoch_(self):
        if getattr(self, "_last_epoch_samples_count", 0) == 0:
            return np.nan
        return self.active_samples_in_epoch_ / self._last_epoch_samples_count

    @property
    def mean_update_norm_in_epoch_(self):
        if self.updates_in_epoch_ == 0:
            return 0.0
        return self.update_norm_sum_in_epoch_ / self.updates_in_epoch_

    def get_epoch_counters(self):
        """Return a dict snapshot of the epoch-level and total counters."""
        check_is_fitted(self)
        return {
            "UpdatesInEpoch": self.updates_in_epoch_,
            "ConstraintChecksInEpoch": self.constraint_checks_in_epoch_,
            "UpdateRate": self.update_rate_in_epoch_,
            "ActiveSamplesInEpoch": self.active_samples_in_epoch_,
            "ActiveSampleRate": self.active_sample_rate_in_epoch_,
            "UpdateNormSumInEpoch": self.update_norm_sum_in_epoch_,
            "UpdateNormMaxInEpoch": self.update_norm_max_in_epoch_,
            "UpdateNormMeanInEpoch": self.mean_update_norm_in_epoch_,
            "TotalUpdates": self.total_updates_,
            "TotalConstraintChecks": self.total_constraint_checks_,
            "TotalActiveSamples": self.total_active_samples_,
            "TotalUpdateNormSum": self.total_update_norm_sum_,
            "TotalUpdateNormMax": self.total_update_norm_max_,
        }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _check_binary_classes(classes):
        """Validate a 2-class label set, raising sklearn-conventional messages."""
        classes = np.unique(np.asarray(classes))
        if len(classes) == 1:
            raise ValueError(
                "This solver needs samples of at least 2 classes in the "
                f"data, but the data contains only one class: {classes[0]!r}"
            )
        if len(classes) > 2:
            raise ValueError(
                "Only binary classification is supported. The type of "
                f"the target is {type_of_target(classes)}."
            )
        return classes

    def _initialize(self, classes):
        classes = self._check_binary_classes(classes)
        self.classes_ = classes
        self.neg_label_, self.pos_label_ = classes[0], classes[1]

        features_count = self.n_features_in_ + (1 if self.fit_bias else 0)

        self._rng = np.random.default_rng(self.random_state)
        self.kappa_hk_ = self._rng.random(features_count) * 0.01
        self.a_hkm_ = np.zeros((features_count, features_count), dtype=float)
        self.a_hm_ = np.zeros(features_count, dtype=float)
        self.m_ = -1

        self._reset_epoch_counters()
        self._reset_total_counters()

        return self

    # ------------------------------------------------------------------
    # Fit / partial_fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X, y = validate_data(self, X, y, reset=True)
        check_classification_targets(y)

        self._initialize(classes=np.unique(y))

        s = self._prepare_labels(y)

        if self.fit_bias:
            X = self._append_dummy_feature(X)

        self._main_loop(X, s)
        return self

    def partial_fit(self, X, y, classes=None):
        first_call = not hasattr(self, "kappa_hk_")
        X, y = validate_data(self, X, y, reset=first_call)

        if first_call:
            if classes is None:
                raise ValueError(
                    "For classification, provide `classes` on the first "
                    "call to partial_fit()."
                )
            self._initialize(classes=classes)
        elif classes is not None:
            classes = np.unique(np.asarray(classes))
            if not np.array_equal(classes, self.classes_):
                raise ValueError(
                    f"`classes={classes!r}` passed to partial_fit() does "
                    f"not match the classes seen on the first call, "
                    f"`classes_={self.classes_!r}`."
                )

        s = self._prepare_labels(y)

        if self.fit_bias:
            X = self._append_dummy_feature(X)

        n = X.shape[0]
        self._last_epoch_samples_count = n
        self._reset_epoch_counters()

        if self.shuffle and n > 1:
            idx = self._rng.permutation(n)
        else:
            idx = range(n)

        for i in idx:
            self._one_sample_step(X[i : i + 1], float(s[i]))

        return self

    def _main_loop(self, X, s):
        self.m_ = -1
        samples_count = X.shape[0]
        features_size = X.shape[1]
        self._last_epoch_samples_count = samples_count

        for _ in range(self.epochs_count):
            self._reset_epoch_counters()

            if self.shuffle and samples_count > 1:
                perm = self._rng.permutation(samples_count)
            else:
                perm = np.arange(samples_count)

            Xp = X[perm]
            sp = s[perm]

            if self.mode != "cumulative":
                self.m_ = -1

            for i in range(samples_count):
                self._one_sample_step(
                    Xp[i].reshape(1, features_size),
                    float(sp[i]),
                )

    # ------------------------------------------------------------------
    # One sample update
    # ------------------------------------------------------------------

    def _one_sample_step(self, x, s):
        features_count = x.shape[1]

        if self.mode == "single":
            self.m_ = 0
        else:
            self.m_ += 1

        m = self.m_

        xxT = np.dot(
            x.reshape((features_count, 1)),
            x.reshape((1, features_count)),
        )

        if self.lam != 0.0:
            reg = self.lam * np.eye(features_count, dtype=float)
            if self.fit_bias and (not self.regularize_bias):
                reg[-1, -1] = 0.0
            xxT = xxT + reg

        self.a_hkm_ = (m / (m + 1)) * self.a_hkm_ + (1 / (m + 1)) * xxT
        self.a_hm_ = (m / (m + 1)) * self.a_hm_ + (1 / (m + 1)) * (
            s * x.reshape(features_count)
        )

        sample_had_update = False

        for h in self._rng.permutation(features_count).tolist():
            self.constraint_checks_in_epoch_ += 1
            self.total_constraint_checks_ += 1

            delta_hm = self._calculate_next_delta(h)

            if abs(delta_hm) >= self.delta_m:
                did_update = self._update_kappa(delta_hm, h)
                if did_update:
                    sample_had_update = True

        if sample_had_update:
            self.active_samples_in_epoch_ += 1
            self.total_active_samples_ += 1

    def _calculate_next_delta(self, h):
        return float(np.dot(self.a_hkm_[h], self.kappa_hk_) - self.a_hm_[h])

    def _update_kappa(self, delta_hm, h):
        denom = float(np.dot(self.a_hkm_[h], self.a_hkm_[h].T))

        if denom <= 0:
            return False

        coeff = delta_hm / denom
        update_vec = coeff * self.a_hkm_[h]
        update_norm = float(np.linalg.norm(update_vec))

        self.kappa_hk_ = self.kappa_hk_ - update_vec

        self.updates_in_epoch_ += 1
        self.total_updates_ += 1

        self.update_norm_sum_in_epoch_ += update_norm
        self.update_norm_max_in_epoch_ = max(
            self.update_norm_max_in_epoch_,
            update_norm,
        )

        self.total_update_norm_sum_ += update_norm
        self.total_update_norm_max_ = max(
            self.total_update_norm_max_,
            update_norm,
        )

        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.fit_bias:
            X = self._append_dummy_feature(X)

        return (X @ self.kappa_hk_).reshape(-1)

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= self.thresh, self.pos_label_, self.neg_label_)

    # ------------------------------------------------------------------
    # Labels / bias
    # ------------------------------------------------------------------

    def _prepare_labels(self, y):
        y = np.asarray(y)
        unknown = np.setdiff1d(np.unique(y), self.classes_)
        if unknown.size:
            raise ValueError(
                f"y contains labels not in classes_={self.classes_!r}: "
                f"{unknown.tolist()!r}"
            )
        return np.where(y == self.pos_label_, 1.0, -1.0).astype(float)

    def _append_dummy_feature(self, X):
        X = np.asarray(X)
        dummy_feature = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([X, dummy_feature])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags
