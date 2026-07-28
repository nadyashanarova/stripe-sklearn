"""Linear C-system Stripe classifier (Yakubovich).

Ported from ``references/linear_reference.py`` (class
``StripeCSystemCorrected``). See ``IMPLEMENTATION_PLAN.md`` and
``MATH_AUDIT.md`` at the repository root for the full analysis this
port is based on. Every approved deviation from the reference
implementation is an sklearn/API-compliance change only (constructor
shape, input validation, dead-code removal); the projection
mathematics -- the pointwise residual, the three-zone multizone
correction ``Psi_mz`` (paper eq. in Section 3.6), the ``eps``/``delta``/
``beta`` boundaries, and the decay-before-residual regularization order
(``MATH_AUDIT.md`` finding M7, confirmed correct by the research
supervisor) -- is preserved exactly.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class StripeCClassifier(ClassifierMixin, BaseEstimator):
    """Yakubovich Stripe / C-system linear binary classifier.

    Learning is posed as a system of pointwise hyperslab constraints on
    the prediction error, one per observation (paper Section 3.6/4.2).
    A parameter correction is applied only when a sample's residual
    violates its tolerance ``eps``, via a three-zone multizone
    correction function ``Psi_mz`` with buffer width ``delta = eps *
    delta_ratio`` and relaxation ``beta``: no update below ``eps``, a
    relaxed correction in ``[eps, eps + delta)``, a reflection-style
    correction in ``[eps + delta, 2*eps)``, and a full projection onto
    the central hyperplane for violations at or above ``2*eps``.

    Parameters
    ----------
    eps : float, default=0.1
        Tolerance defining the width of the admissible hyperslab around
        each pointwise constraint.
    delta_ratio : float, default=0.05
        Buffer-zone width as a fraction of ``eps``; see the ``delta``
        property.
    beta : float, default=1.0
        Relaxation parameter for the buffer-zone correction.
    epochs_count : int, default=1
        Number of passes over the training data performed by ``fit``
        (implemented as ``epochs_count`` calls to ``partial_fit``, each
        with its own fresh shuffle, exactly mirroring the reference).
    fit_bias : bool, default=True
        Whether to append a constant feature (bias/intercept) to the
        input.
    lam : float, default=0.0
        Multiplicative weight-decay strength, applied every sample
        *before* the residual is evaluated (see ``MATH_AUDIT.md``
        finding M7: this order was confirmed correct against the
        reference implementation and the kernel C-system, resolving an
        imprecise description in the paper's Appendix B.2 text).
    regularize_bias : bool, default=False
        Whether the bias coordinate (when ``fit_bias=True``) is
        included in the weight decay.
    shuffle : bool, default=True
        Whether to shuffle sample order using the estimator's random
        state.
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
    init_mode : {"zeros", "random"}, default="zeros"
        Weight initialization strategy. ``"zeros"`` matches the paper's
        worked example (``theta_1 = 0``); ``"random"`` draws from
        ``Uniform[0, init_scale)``.
    init_scale : float, default=0.01
        Scale of the random initialization when ``init_mode="random"``.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The two class labels seen during ``fit``, sorted ascending;
        ``classes_[0]`` is the negative class, ``classes_[1]`` the
        positive class.
    n_features_in_ : int
        Number of features seen during ``fit`` (excluding the bias
        feature, if any).
    weights_ : ndarray of shape (n_features_in_ + int(fit_bias),)
        Learned parameter vector.
    updates_in_epoch_, total_updates_ : int
        Number of corrections in the most recent ``partial_fit`` call,
        and cumulatively since the last (re)initialization.
    constraint_checks_in_epoch_, total_constraint_checks_ : int
        Number of pointwise constraint evaluations.
    zone0_, zone1_, zone2_, zone3_ : int
        Number of samples falling in each zone of ``Psi_mz`` during the
        most recent ``partial_fit`` call (zone0: no update, |eta| < eps;
        zone1: full projection, |eta| >= 2*eps; zone2: reflection zone;
        zone3: buffer zone) -- reset every ``partial_fit`` call, exactly
        as in the reference (there is no cumulative ``total_zoneX_``
        counterpart, matching the reference implementation).
    """

    def __init__(
        self,
        eps=0.1,
        delta_ratio=0.05,
        beta=1.0,
        epochs_count=1,
        fit_bias=True,
        lam=0.0,
        regularize_bias=False,
        shuffle=True,
        random_state=None,
        thresh=0.0,
        init_mode="zeros",
        init_scale=0.01,
    ):
        self.eps = eps
        self.delta_ratio = delta_ratio
        self.beta = beta
        self.epochs_count = epochs_count
        self.fit_bias = fit_bias
        self.lam = lam
        self.regularize_bias = regularize_bias
        self.shuffle = shuffle
        self.random_state = random_state
        self.thresh = thresh
        self.init_mode = init_mode
        self.init_scale = init_scale

    @property
    def delta(self):
        return float(self.eps) * float(self.delta_ratio)

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

        self._rng = np.random.default_rng(self.random_state)

        features_count = self.n_features_in_ + (1 if self.fit_bias else 0)
        self._initialize_weights(features_count)

        self.updates_in_epoch_ = 0
        self.total_updates_ = 0
        self.constraint_checks_in_epoch_ = 0
        self.total_constraint_checks_ = 0
        self.zone0_ = self.zone1_ = self.zone2_ = self.zone3_ = 0

        return self

    def _initialize_weights(self, features_count):
        if self.init_mode == "zeros":
            self.weights_ = np.zeros(features_count, dtype=float)
        elif self.init_mode == "random":
            self.weights_ = self._rng.random(features_count) * self.init_scale
        else:
            raise ValueError("init_mode must be 'zeros' or 'random'")

    # ------------------------------------------------------------------
    # Fit / partial_fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X, y = validate_data(self, X, y, reset=True)
        check_classification_targets(y)

        classes = np.unique(y)
        self._initialize(classes=classes)

        for _ in range(self.epochs_count):
            self.partial_fit(X, y, classes=classes)

        return self

    def partial_fit(self, X, y, classes=None):
        first_call = not hasattr(self, "weights_")
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
        idx = self._rng.permutation(n) if (self.shuffle and n > 1) else range(n)

        self.updates_in_epoch_ = 0
        self.constraint_checks_in_epoch_ = 0
        self.zone0_ = self.zone1_ = self.zone2_ = self.zone3_ = 0

        for i in idx:
            self.constraint_checks_in_epoch_ += 1
            self.total_constraint_checks_ += 1
            self._one_sample_step(X[i], float(s[i]))

        return self

    # ------------------------------------------------------------------
    # One sample update
    # ------------------------------------------------------------------

    def _one_sample_step(self, x, s):
        if self.lam > 0.0:
            decay = 1.0 - self.lam
            if self.fit_bias and (not self.regularize_bias):
                self.weights_[:-1] *= decay
            else:
                self.weights_ *= decay

        pred = float(np.dot(self.weights_, x))
        eta = pred - float(s)
        abs_eta = abs(eta)

        if abs_eta < self.eps:
            self.zone0_ += 1
            return

        norm_sq = float(np.dot(x, x))
        if norm_sq < 1e-12:
            return

        inv_norm = 1.0 / norm_sq
        sign_eta = np.sign(eta) if eta != 0.0 else 1.0

        if abs_eta >= 2.0 * self.eps:
            self.zone1_ += 1
            zeta = -eta * inv_norm
        elif (self.eps + self.delta) <= abs_eta < 2.0 * self.eps:
            self.zone2_ += 1
            zeta = 2.0 * inv_norm * (self.eps * sign_eta - eta)
        else:
            self.zone3_ += 1
            zeta = self.beta * inv_norm * (self.eps * sign_eta - eta)

        if zeta != 0.0:
            self.weights_ += zeta * x
            self.updates_in_epoch_ += 1
            self.total_updates_ += 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.fit_bias:
            X = self._append_dummy_feature(X)

        return X.dot(self.weights_).reshape(-1)

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
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([X, ones])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags
