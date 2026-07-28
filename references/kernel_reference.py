import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels


class StripeKernel(ClassifierMixin, BaseEstimator):
    """
    Stripe / L-system (kernelized) in an epoch-wise regime.

    PATCH:
      - updates_in_epoch_ (resets each epoch)
      - total_updates_ (cumulative)
      - FIX: reset updates_in_epoch_ even if M==0
      - FIX: protect partial_fit on fresh model when X/y are None
    """

    def __init__(self, delta_m=1e-10, lam=0.01, kernel="rbf", gamma="scale", random_state=None):
        self.delta_m = float(delta_m)
        self.lam = float(lam)
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self._system_dirty = True

    def _get_kernel_func(self, X1, X2):
        if self.kernel == "linear":
            return linear_kernel(X1, X2)
        return rbf_kernel(X1, X2, gamma=self.gamma_)

    def _init_state(self, n_features: int):
        self.n_features_in_ = n_features
        self.X_train_ = np.empty((0, n_features))
        self.y_train_internal_ = np.empty(0)  # +/-1
        self.alpha_ = np.empty(0)

        self.m_ = 0
        self.K_all_ = np.empty((0, 0))
        self.B_unnormalized_ = np.empty((0, 0))
        self.c_unnormalized_ = np.empty(0)

        self._rng = np.random.default_rng(self.random_state)
        self._is_fitted = True

        self.A_cached_ = None
        self.c_cached_ = None
        self.row_norms_sq_ = None
        self._system_dirty = True

        # counters
        self.updates_in_epoch_ = 0
        self.total_updates_ = 0

    def fit(self, X, y, classes=None, epochs_count=0):
        X, y = check_X_y(X, y)

        self.classes_ = classes if classes is not None else unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("StripeKernel currently supports binary classification only.")
        self.neg_label_, self.pos_label_ = self.classes_[0], self.classes_[1]

        n_features = X.shape[1]
        if isinstance(self.gamma, (float, int)):
            self.gamma_ = float(self.gamma)
        elif self.gamma == "auto":
            self.gamma_ = 1.0 / n_features
        else:  # 'scale'
            var = float(X.var())
            self.gamma_ = 1.0 / (n_features * var) if var > 1e-8 else 1.0 / n_features

        self._init_state(n_features)

        # accumulate system
        self.partial_fit(X, y, classes=self.classes_, accumulate=True)

        # epochs
        for _ in range(int(epochs_count)):
            self.partial_fit()

        return self

    def partial_fit(self, X=None, y=None, classes=None, accumulate=False):
        if not hasattr(self, "_is_fitted"):
            if X is None or y is None:
                raise ValueError("Call fit(X,y) first or partial_fit(X,y,accumulate=True) first.")
            return self.fit(X, y, classes=classes, epochs_count=0)

        if accumulate:
            if X is None or y is None:
                raise ValueError("accumulate=True requires X and y.")
            X, y = check_X_y(X, y)
            s_int = np.where(y == self.pos_label_, 1.0, -1.0)
            for i in range(X.shape[0]):
                self._accumulate_one(X[i:i + 1], float(s_int[i]))
            return self

        self._run_one_epoch_stripe()
        return self

    def _accumulate_one(self, x, s: float):
        if self.m_ == 0:
            k_new = self._get_kernel_func(x, x).item()
            self.K_all_ = np.array([[k_new]])
            self.c_unnormalized_ = np.array([s * k_new])
            self.B_unnormalized_ = np.array([[k_new ** 2]])
        else:
            k_vec = self._get_kernel_func(x, self.X_train_).flatten()
            k_new = self._get_kernel_func(x, x).item()

            K_m_v = self.K_all_.T @ k_vec
            v_vT = np.outer(k_vec, k_vec)

            self.B_unnormalized_ = np.block([
                [self.B_unnormalized_ + v_vT, (K_m_v + k_vec * k_new)[:, None]],
                [(K_m_v + k_vec * k_new)[None, :], np.array([[np.dot(k_vec, k_vec) + k_new ** 2]])]
            ])

            self.K_all_ = np.block([
                [self.K_all_, k_vec[:, None]],
                [k_vec[None, :], np.array([[k_new]])]
            ])

            self.c_unnormalized_ = np.append(
                self.c_unnormalized_ + s * k_vec,
                np.dot(self.y_train_internal_, k_vec) + s * k_new
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

    def decision_function(self, X):
        X = check_array(X)
        return self._get_kernel_func(X, self.X_train_).dot(self.alpha_)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, self.pos_label_, self.neg_label_)


class CSystemKernelEpochWise(ClassifierMixin, BaseEstimator):
    """
    Epoch-wise C-system kernel (Yakubovich-style, kernelized).

    PATCH:
      - updates_in_epoch_ (resets each epoch)
      - total_updates_ (cumulative)
      - FIX: reset updates_in_epoch_ even if M==0 (avoid stale values)
    """

    def __init__(self, eps=0.1, delta_ratio=0.5, beta_param=1.0,
                 lam=0.0, delta_m=1e-12, kernel='rbf', gamma='scale', random_state=None):
        self.eps = float(eps)
        self.delta_ratio = float(delta_ratio)
        self.beta_param = float(beta_param)
        self.lam = float(lam)
        self.delta_m = float(delta_m)
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state

    def _get_kernel_func(self, X1, X2):
        return rbf_kernel(X1, X2, gamma=self.gamma_) if self.kernel == 'rbf' else linear_kernel(X1, X2)

    def _init_state(self, n_features: int):
        self.n_features_in_ = n_features
        self.X_train_ = np.empty((0, n_features))
        self.y_train_signed_ = np.empty(0)
        self.alpha_ = np.empty(0)
        self.m_ = 0

        self.K_all_ = np.empty((0, 0))
        self.norms_sq_C_ = np.empty(0)

        self._rng = np.random.default_rng(self.random_state)
        self._is_fitted = True

        # counters
        self.updates_in_epoch_ = 0
        self.total_updates_ = 0

    def fit(self, X, y, classes=None, epochs_count=0):
        X, y = check_X_y(X, y)
        self.classes_ = classes if classes is not None else unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("CSystemKernelEpochWise supports binary classification only.")

        self.neg_label_, self.pos_label_ = self.classes_[0], self.classes_[1]
        self.delta_ = self.eps * self.delta_ratio

        n_features = X.shape[1]
        if isinstance(self.gamma, (float, int)):
            self.gamma_ = float(self.gamma)
        elif self.gamma == 'auto':
            self.gamma_ = 1.0 / n_features
        else:  # 'scale'
            var = float(X.var())
            self.gamma_ = 1.0 / (n_features * var) if var > 1e-8 else 1.0 / n_features

        self._init_state(n_features)

        # accumulate once
        self.partial_fit(X, y, classes=self.classes_, accumulate=True)

        # epochs
        for _ in range(int(epochs_count)):
            self.partial_fit()

        return self

    def partial_fit(self, X=None, y=None, classes=None, accumulate=False):
        if not hasattr(self, "_is_fitted"):
            if X is None or y is None:
                raise ValueError("Call fit(X,y) first or use partial_fit(X,y,accumulate=True) first.")
            return self.fit(X, y, classes=classes, epochs_count=0)

        if accumulate:
            if X is None or y is None:
                raise ValueError("accumulate=True requires X and y.")
            X, y = check_X_y(X, y)
            y_signed = np.where(y == self.pos_label_, 1.0, -1.0)
            for i in range(X.shape[0]):
                self._accumulate_one(X[i:i+1], float(y_signed[i]))
            return self

        self._run_one_epoch()
        return self

    def _accumulate_one(self, x, y_i: float):
        if self.m_ == 0:
            k_s = self._get_kernel_func(x, x).item()
            self.K_all_ = np.array([[k_s]])
            self.norms_sq_C_ = np.array([k_s ** 2])
        else:
            k_v = self._get_kernel_func(x, self.X_train_).flatten()
            k_s = self._get_kernel_func(x, x).item()

            self.K_all_ = np.block([
                [self.K_all_, k_v[:, None]],
                [k_v[None, :], np.array([[k_s]])]
            ])

            self.norms_sq_C_ = (self.norms_sq_C_ + k_v ** 2)
            self.norms_sq_C_ = np.append(self.norms_sq_C_, np.sum(k_v ** 2) + k_s ** 2)

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
            # regularize BEFORE checking/correcting constraint k
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
            elif (self.eps + self.delta_) <= abs_eta < 2 * self.eps:
                xi = 2 * (self.eps * np.sign(eta_k) - eta_k) / norm_sq
            elif self.eps <= abs_eta < (self.eps + self.delta_):
                xi = self.beta_param * (self.eps * np.sign(eta_k) - eta_k) / norm_sq

            if xi != 0.0:
                self.alpha_ -= xi * self.K_all_[k]
                self.updates_in_epoch_ += 1
                self.total_updates_ += 1

    def decision_function(self, X):
        X = check_array(X)
        return self._get_kernel_func(X, self.X_train_).dot(self.alpha_)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, self.pos_label_, self.neg_label_)