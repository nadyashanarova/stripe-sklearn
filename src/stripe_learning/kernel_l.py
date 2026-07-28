"""Kernel L-system Stripe classifier (Yakubovich).

Ported from ``references/kernel_reference.py`` (class ``StripeKernel``).
See ``IMPLEMENTATION_PLAN.md`` and ``MATH_AUDIT.md`` at the repository
root for the full analysis this port is based on. The projection
mathematics -- the exact incremental Gram-matrix accumulation
(``MATH_AUDIT.md`` Section 3.3, verified by hand-derivation to equal a
from-scratch recomputation of ``K^T K`` / ``K^T y``, not an
approximation), the regularized system ``A = B/M + lam*I``, and the
basic-Stripe projection step with ``delta_m`` playing the role of
``epsilon`` (``MATH_AUDIT.md`` finding F1/M10) -- is preserved exactly.

Approved deviations from the reference (see ``IMPLEMENTATION_PLAN.md``
sections 5.1, 5.2, 5.10, 5.11) are sklearn/API-compliance changes only:

- ``epochs_count`` moves from a ``fit()`` keyword argument (reference
  default ``0``, which silently produced an untrained, all-zero model)
  to a constructor parameter defaulting to ``1``, matching the linear
  estimators and making it tunable via ``GridSearchCV``/``clone``.
- ``partial_fit(X, y, classes=None)`` always consumes the supplied data
  (accumulating it into the kernel dictionary) and then runs one
  correction epoch, instead of requiring a separate ``accumulate=True``
  flag that a caller could forget, silently leaving new data unused.
- ``thresh`` is added as a constructor parameter (default ``0.0``,
  reproducing the reference's hardcoded threshold) for API consistency
  with ``StripeLClassifier``/``StripeCClassifier``.
- ``__init__`` only assigns constructor arguments, unmodified; all
  validation, RNG construction, and fitted-state initialization moved
  into ``fit``/``partial_fit``.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class StripeLKernelClassifier(ClassifierMixin, BaseEstimator):
    """Yakubovich Stripe / L-system kernelized binary classifier.

    The model is a kernel expansion ``f(x) = sum_k alpha_k K(x_k, x)``
    over the training dictionary (paper Section 5.1-5.2). Learning
    proceeds in two phases: (1) accumulating a Gram-matrix-based
    regularized linear system in the space of expansion coefficients
    ``alpha`` (exactly, via an incremental block-matrix update -- see
    ``MATH_AUDIT.md`` Section 3.3), and (2) running basic-Stripe
    projection passes over that system's rows (paper eq. 6, with
    ``delta_m`` playing the role of ``epsilon`` -- see ``MATH_AUDIT.md``
    finding F1/M10: the improved-Stripe variant, eq. 7, is intentionally
    not implemented, so Theorem 1 does not directly cover this
    estimator).

    Parameters
    ----------
    delta_m : float, default=1e-10
        Update-trigger tolerance for the Stripe projection step; plays
        the role of ``epsilon`` in the paper's basic Stripe update
        (eq. 6). See the class docstring and ``MATH_AUDIT.md`` finding
        F1/M10.
    lam : float, default=0.01
        L2 regularization strength added to the diagonal of the
        Gram-based system (paper eq. 15, ``+ lam * I``).
    kernel : {"rbf", "linear"}, default="rbf"
        Kernel function. Any value other than the literal string
        ``"linear"`` is treated as ``"rbf"`` (matching the reference
        implementation exactly, including the lack of validation for
        unrecognized kernel names). Note the dedicated ``"linear"``
        kernel choice here is a different algorithm from
        ``StripeLClassifier`` and is not a substitute for it (see
        ``CLAUDE.md``).
    gamma : {"scale", "auto"} or float, default="scale"
        RBF kernel coefficient, following the same ``"scale"``/``"auto"``
        semantics as scikit-learn's own RBF-kernel estimators. Any
        string other than the literal ``"auto"`` is treated as
        ``"scale"``, matching the reference implementation exactly.
        Ignored when ``kernel="linear"``.
    epochs_count : int, default=1
        Number of Stripe correction epochs run over the accumulated
        dictionary by ``fit`` (see the module docstring: this was a
        ``fit()``-time keyword defaulting to ``0`` in the reference,
        moved to the constructor and defaulted to ``1`` for sklearn
        compatibility).
    thresh : float, default=0.0
        Decision threshold ``tau`` applied to ``decision_function`` in
        ``predict`` (paper eq. 1). Added as a constructor parameter for
        consistency with ``StripeLClassifier``/``StripeCClassifier``;
        the reference hardcoded this to ``0.0`` inline.
    random_state : int, array-like of ints, numpy.random.SeedSequence, \
            numpy.random.BitGenerator, numpy.random.Generator, or None, \
            default=None
        Controls the per-epoch row-shuffling randomness. Passed
        straight through to ``numpy.random.default_rng``; see
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
        Number of coefficient corrections in the most recent Stripe
        epoch, and cumulatively since the last (re)initialization.
    """

    def __init__(
        self,
        delta_m=1e-10,
        lam=0.01,
        kernel="rbf",
        gamma="scale",
        epochs_count=1,
        thresh=0.0,
        random_state=None,
    ):
        self.delta_m = delta_m
        self.lam = lam
        self.kernel = kernel
        self.gamma = gamma
        self.epochs_count = epochs_count
        self.thresh = thresh
        self.random_state = random_state

    def _get_kernel_func(self, X1, X2):
        if self.kernel == "linear":
            return linear_kernel(X1, X2)
        return rbf_kernel(X1, X2, gamma=self.gamma_)

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
        self.y_train_internal_ = np.empty(0)  # +/-1
        self.alpha_ = np.empty(0)

        self.m_ = 0
        self.K_all_ = np.empty((0, 0))
        self.B_unnormalized_ = np.empty((0, 0))
        self.c_unnormalized_ = np.empty(0)

        self._rng = np.random.default_rng(self.random_state)

        self.A_cached_ = None
        self.c_cached_ = None
        self.row_norms_sq_ = None
        self._system_dirty = True

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
            self._run_one_epoch_stripe()

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
        self._run_one_epoch_stripe()
        return self

    def _accumulate_batch(self, X, y):
        s = self._prepare_labels(y)
        for i in range(X.shape[0]):
            self._accumulate_one(X[i : i + 1], float(s[i]))

    def _accumulate_one(self, x, s):
        if self.m_ == 0:
            k_new = self._get_kernel_func(x, x).item()
            self.K_all_ = np.array([[k_new]])
            self.c_unnormalized_ = np.array([s * k_new])
            self.B_unnormalized_ = np.array([[k_new**2]])
        else:
            k_vec = self._get_kernel_func(x, self.X_train_).flatten()
            k_new = self._get_kernel_func(x, x).item()

            K_m_v = self.K_all_.T @ k_vec
            v_vT = np.outer(k_vec, k_vec)

            self.B_unnormalized_ = np.block(
                [
                    [self.B_unnormalized_ + v_vT, (K_m_v + k_vec * k_new)[:, None]],
                    [
                        (K_m_v + k_vec * k_new)[None, :],
                        np.array([[np.dot(k_vec, k_vec) + k_new**2]]),
                    ],
                ]
            )

            self.K_all_ = np.block(
                [
                    [self.K_all_, k_vec[:, None]],
                    [k_vec[None, :], np.array([[k_new]])],
                ]
            )

            self.c_unnormalized_ = np.append(
                self.c_unnormalized_ + s * k_vec,
                np.dot(self.y_train_internal_, k_vec) + s * k_new,
            )

        self.X_train_ = np.vstack([self.X_train_, x])
        self.y_train_internal_ = np.append(self.y_train_internal_, s)
        self.alpha_ = np.append(self.alpha_, 0.0)
        self.m_ += 1
        self._system_dirty = True

    def _build_cached_system_if_needed(self):
        M = self.m_
        if M == 0:
            self.A_cached_ = None
            self.c_cached_ = None
            self.row_norms_sq_ = None
            self._system_dirty = False
            return

        if not self._system_dirty:
            return

        A = self.B_unnormalized_ / M
        c = self.c_unnormalized_ / M

        if self.lam > 0:
            A = A + np.eye(M) * self.lam

        self.A_cached_ = A
        self.c_cached_ = c
        self.row_norms_sq_ = np.einsum("ij,ij->i", A, A)
        self._system_dirty = False

    def _run_one_epoch_stripe(self):
        # always reset per-epoch counter
        self.updates_in_epoch_ = 0
        M = self.m_
        if M == 0:
            return

        self._build_cached_system_if_needed()
        A = self.A_cached_
        c = self.c_cached_
        row_norms_sq = self.row_norms_sq_

        for p in self._rng.permutation(M):
            delta = float(np.dot(A[p, :], self.alpha_) - c[p])
            if abs(delta) >= self.delta_m:
                n_sq = float(row_norms_sq[p])
                if n_sq > 1e-15:
                    self.alpha_ -= (delta / n_sq) * A[p, :]
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
