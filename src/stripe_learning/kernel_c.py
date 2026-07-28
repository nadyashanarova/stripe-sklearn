"""Kernel C-system Stripe classifier (Yakubovich).

Ported from ``references/kernel_reference.py`` (class
``CSystemKernelEpochWise``). See ``IMPLEMENTATION_PLAN.md`` and
``MATH_AUDIT.md`` at the repository root for the full analysis this
port is based on. The projection mathematics -- the exact incremental
Gram-matrix/row-norm accumulation (``MATH_AUDIT.md`` Section 4,
verified by hand-derivation), the three-zone multizone correction
``Psi_mz`` in kernel-coefficient space, the ``y - pred`` residual sign
convention (deliberately *not* harmonized with the linear C-system's
``pred - y`` convention -- see ``MATH_AUDIT.md`` finding M11 and
``IMPLEMENTATION_PLAN.md`` section 5.7), and the decay-applied-before-
every-constraint regularization order (which already matched both the
reference code and the paper's Appendix B.4 text -- unlike the linear
C-system, no paper/code discrepancy exists here) -- is preserved
exactly.

Approved deviations from the reference (see ``IMPLEMENTATION_PLAN.md``
sections 5.1, 5.2, 5.10, 5.11) mirror ``StripeLKernelClassifier``'s:

- ``epochs_count`` moves from a ``fit()`` keyword argument (reference
  default ``0``) to a constructor parameter defaulting to ``1``.
- ``partial_fit(X, y, classes=None)`` always consumes the supplied data
  and then runs one correction epoch, instead of requiring a separate
  ``accumulate=True`` flag.
- ``thresh`` is added as a constructor parameter (default ``0.0``).
- ``__init__`` only assigns constructor arguments, unmodified.
- ``delta`` is exposed as a read-only property (``eps * delta_ratio``),
  matching ``StripeCClassifier``, instead of a fitted attribute
  (``delta_``) recomputed once inside ``fit()`` as in the reference;
  the value is identical either way, this only changes when it is
  computed (on demand vs. cached at fit time).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class StripeCKernelClassifier(ClassifierMixin, BaseEstimator):
    """Yakubovich Stripe / C-system kernelized binary classifier.

    Like ``StripeLKernelClassifier``, the model is a kernel expansion
    ``f(x) = sum_k alpha_k K(x_k, x)`` over the training dictionary, but
    learning follows the pointwise multizone correction ``Psi_mz`` of
    the C-system (paper Section 5.3) rather than the L-system's
    relaxed-stationarity projection: each accumulated point is one
    pointwise constraint, processed with the same three-zone correction
    as ``StripeCClassifier`` (no update below ``eps``, a relaxed
    correction in ``[eps, eps + delta)``, a reflection-style correction
    in ``[eps + delta, 2*eps)``, and a full projection at or above
    ``2*eps``), operating on rows of the kernel Gram matrix instead of
    raw feature vectors.

    Parameters
    ----------
    eps : float, default=0.1
        Tolerance defining the width of the admissible hyperslab around
        each pointwise constraint.
    delta_ratio : float, default=0.5
        Buffer-zone width as a fraction of ``eps``; see the ``delta``
        property. Note the reference default for this parameter differs
        10x between the linear (0.05) and kernel (0.5) C-systems; both
        defaults are preserved exactly as in their respective
        references (see ``IMPLEMENTATION_PLAN.md`` section 5.12).
    beta_param : float, default=1.0
        Relaxation parameter for the buffer-zone correction.
    lam : float, default=0.0
        Multiplicative weight-decay strength, applied to the full
        ``alpha_`` vector before *every* constraint evaluation within
        an epoch (not once per epoch) -- see the module docstring.
    delta_m : float, default=1e-12
        Degenerate-row-norm safety floor: a constraint is skipped
        entirely (no update, no counter increment) if its Gram row norm
        squared is below this value. Unlike the L-system estimators,
        this is *not* a Stripe update-trigger tolerance (that role is
        played by ``eps``/``delta``/``beta_param`` here) -- see
        ``MATH_AUDIT.md`` finding on ``delta_m``'s dual meaning across
        the L- and C-systems (``IMPLEMENTATION_PLAN.md`` section 5.6:
        name preserved unchanged for compatibility, meaning documented
        here instead).
    kernel : {"rbf", "linear"}, default="rbf"
        Kernel function. Any value other than the literal string
        ``"rbf"`` is treated as ``"linear"`` -- note this fallback
        direction is the *opposite* of ``StripeLKernelClassifier``,
        where any value other than ``"linear"`` is treated as
        ``"rbf"``. Both directions are preserved exactly as found in
        their respective reference classes; this asymmetry is a
        reference-implementation quirk, not something introduced here.
    gamma : {"scale", "auto"} or float, default="scale"
        RBF kernel coefficient; see ``StripeLKernelClassifier``.
        Ignored when ``kernel="linear"``.
    epochs_count : int, default=1
        Number of Stripe correction epochs run over the accumulated
        dictionary by ``fit`` (a ``fit()``-time keyword defaulting to
        ``0`` in the reference; moved to the constructor and defaulted
        to ``1`` for sklearn compatibility -- see the module docstring).
    thresh : float, default=0.0
        Decision threshold ``tau`` applied to ``decision_function`` in
        ``predict`` (paper eq. 1); the reference hardcoded this to
        ``0.0`` inline.
    random_state : int, array-like of ints, numpy.random.SeedSequence, \
            numpy.random.BitGenerator, numpy.random.Generator, or None, \
            default=None
        Controls the per-epoch row-shuffling randomness. See
        ``StripeLClassifier``'s docstring for the exact accepted-type
        caveat (this estimator does not use
        ``sklearn.utils.check_random_state``). The random generator is
        reseeded at the start of every ``fit`` call (and on the first
        ``partial_fit`` call), so repeated ``fit`` calls with a fixed
        ``random_state`` are reproducible.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The two class labels seen during ``fit``, sorted ascending.
    n_features_in_ : int
        Number of features seen during ``fit``.
    gamma_ : float
        The RBF gamma value actually used, resolved from ``gamma``.
    X_train_ : ndarray of shape (n_dictionary_points, n_features_in_)
        The kernel dictionary (support points) accumulated so far.
    alpha_ : ndarray of shape (n_dictionary_points,)
        Learned kernel expansion coefficients.
    m_ : int
        Number of points currently in the dictionary.
    updates_in_epoch_, total_updates_ : int
        Number of corrections in the most recent Stripe epoch, and
        cumulatively since the last (re)initialization. Unlike
        ``StripeCClassifier``, there are no per-zone ``zone0_``..
        ``zone3_`` counters here -- the reference does not track them
        for the kernel C-system either.
    """

    def __init__(
        self,
        eps=0.1,
        delta_ratio=0.5,
        beta_param=1.0,
        lam=0.0,
        delta_m=1e-12,
        kernel="rbf",
        gamma="scale",
        epochs_count=1,
        thresh=0.0,
        random_state=None,
    ):
        self.eps = eps
        self.delta_ratio = delta_ratio
        self.beta_param = beta_param
        self.lam = lam
        self.delta_m = delta_m
        self.kernel = kernel
        self.gamma = gamma
        self.epochs_count = epochs_count
        self.thresh = thresh
        self.random_state = random_state

    @property
    def delta(self):
        return float(self.eps) * float(self.delta_ratio)

    def _get_kernel_func(self, X1, X2):
        # Note the fallback direction: unrecognized `kernel` values fall
        # back to "linear" here, but to "rbf" in StripeLKernelClassifier
        # -- both match their respective reference implementations
        # exactly (see the class docstring).
        return (
            rbf_kernel(X1, X2, gamma=self.gamma_)
            if self.kernel == "rbf"
            else linear_kernel(X1, X2)
        )

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

    def _initialize(self, X, classes):
        classes = self._check_binary_classes(classes)
        self.classes_ = classes
        self.neg_label_, self.pos_label_ = classes[0], classes[1]

        n_features = self.n_features_in_
        if isinstance(self.gamma, (float, int)):
            self.gamma_ = float(self.gamma)
        elif self.gamma == "auto":
            self.gamma_ = 1.0 / n_features
        else:  # 'scale'
            var = float(X.var())
            self.gamma_ = 1.0 / (n_features * var) if var > 1e-8 else 1.0 / n_features

        self.X_train_ = np.empty((0, n_features))
        self.y_train_signed_ = np.empty(0)
        self.alpha_ = np.empty(0)
        self.m_ = 0

        self.K_all_ = np.empty((0, 0))
        self.norms_sq_C_ = np.empty(0)

        self._rng = np.random.default_rng(self.random_state)

        # counters
        self.updates_in_epoch_ = 0
        self.total_updates_ = 0

        return self

    # ------------------------------------------------------------------
    # Fit / partial_fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X, y = validate_data(self, X, y, reset=True)
        check_classification_targets(y)

        classes = np.unique(y)
        self._initialize(X, classes)
        self._accumulate_batch(X, y)

        for _ in range(self.epochs_count):
            self._run_one_epoch()

        return self

    def partial_fit(self, X, y, classes=None):
        first_call = not hasattr(self, "alpha_")
        X, y = validate_data(self, X, y, reset=first_call)

        if first_call:
            if classes is None:
                raise ValueError(
                    "For classification, provide `classes` on the first "
                    "call to partial_fit()."
                )
            self._initialize(X, classes)
        elif classes is not None:
            classes = np.unique(np.asarray(classes))
            if not np.array_equal(classes, self.classes_):
                raise ValueError(
                    f"`classes={classes!r}` passed to partial_fit() does "
                    f"not match the classes seen on the first call, "
                    f"`classes_={self.classes_!r}`."
                )

        self._accumulate_batch(X, y)
        self._run_one_epoch()
        return self

    def _accumulate_batch(self, X, y):
        s = self._prepare_labels(y)
        for i in range(X.shape[0]):
            self._accumulate_one(X[i : i + 1], float(s[i]))

    def _accumulate_one(self, x, y_i):
        if self.m_ == 0:
            k_s = self._get_kernel_func(x, x).item()
            self.K_all_ = np.array([[k_s]])
            self.norms_sq_C_ = np.array([k_s**2])
        else:
            k_v = self._get_kernel_func(x, self.X_train_).flatten()
            k_s = self._get_kernel_func(x, x).item()

            self.K_all_ = np.block(
                [
                    [self.K_all_, k_v[:, None]],
                    [k_v[None, :], np.array([[k_s]])],
                ]
            )

            self.norms_sq_C_ = self.norms_sq_C_ + k_v**2
            self.norms_sq_C_ = np.append(self.norms_sq_C_, np.sum(k_v**2) + k_s**2)

        self.X_train_ = np.vstack([self.X_train_, x])
        self.y_train_signed_ = np.append(self.y_train_signed_, y_i)
        self.alpha_ = np.append(self.alpha_, 0.0)
        self.m_ += 1

    def _run_one_epoch(self):
        # always reset per-epoch counter
        self.updates_in_epoch_ = 0
        M = self.m_
        if M == 0:
            return

        decay = (1.0 - self.lam) if self.lam > 0 else 1.0

        for k in self._rng.permutation(M):
            # regularize BEFORE checking/correcting constraint k -- every
            # iteration, not once per epoch (matches the reference and
            # the paper's Appendix B.4 text; see the module docstring)
            if decay != 1.0:
                self.alpha_ *= decay

            eta_k = float(self.y_train_signed_[k] - np.dot(self.alpha_, self.K_all_[k]))
            abs_eta = abs(eta_k)
            norm_sq = float(self.norms_sq_C_[k])

            if norm_sq < self.delta_m:
                continue

            xi = 0.0
            if abs_eta >= 2 * self.eps:
                xi = -eta_k / norm_sq
            elif (self.eps + self.delta) <= abs_eta < 2 * self.eps:
                xi = 2 * (self.eps * np.sign(eta_k) - eta_k) / norm_sq
            elif self.eps <= abs_eta < (self.eps + self.delta):
                xi = self.beta_param * (self.eps * np.sign(eta_k) - eta_k) / norm_sq

            if xi != 0.0:
                self.alpha_ -= xi * self.K_all_[k]
                self.updates_in_epoch_ += 1
                self.total_updates_ += 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self._get_kernel_func(X, self.X_train_).dot(self.alpha_)

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= self.thresh, self.pos_label_, self.neg_label_)

    # ------------------------------------------------------------------
    # Labels
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags
