import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import typing
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
import os
import os, sys, json, warnings, logging, contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score, precision_score, recall_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
print("Base imports OK")

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import warnings
import sys
import os
from time import time

import sys
import os
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import numpy as np
# TabZilla Native
sys.path.append(os.path.join(os.getcwd(), "TabZilla"))
try:
    from tabzilla_data_preprocessing import preprocess_dataset
    from tabzilla_datasets import TabularDataset
    from tabzilla_data_processing import process_data
except ImportError:
    print("[ERROR] Не удалось импортировать модули TabZilla.")

warnings.filterwarnings('ignore')

import random
import typing
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from scipy import sparse

from tqdm import tqdm
import numpy as np
from sklearn.base import BaseEstimator
# ============================================================
# 1) MODEL: Linear C-system
# ============================================================
class StripeCSystemCorrected(ClassifierMixin, BaseEstimator):
    """
    Линейная C-система Якубовича.
    Поддерживает:
      - fit / partial_fit
      - initialize(...)
      - zero/random init
      - updates counters
    """

    def __init__(
        self,
        eps: float = 0.1,
        delta_ratio: float = 0.05,
        beta: float = 1.0,
        epochs_count: int = 1,
        fit_bias: bool = True,
        lam: float = 0.0,
        regularize_bias: bool = False,
        model_type: str = "classification",
        shuffle: bool = True,
        random_state: int | None = None,
        thresh: float = 0.0,
        init_mode: str = "zeros",      # "zeros" | "random"
        init_scale: float = 0.01,
    ):
        self.constraint_checks_in_epoch_ = 0
        self.total_constraint_checks_ = 0
        
        self.eps = float(eps)
        self.delta_ratio = float(delta_ratio)
        self.beta = float(beta)
        self.epochs_count = int(epochs_count)

        self.fit_bias = bool(fit_bias)
        self.lam = float(lam)
        self.regularize_bias = bool(regularize_bias)

        self.model_type = str(model_type)
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.thresh = float(thresh)

        self.init_mode = str(init_mode)
        self.init_scale = float(init_scale)

        self.weights_ = None
        self._rng = np.random.default_rng(self.random_state)

        self.classes_ = None
        self.pos_label_ = None
        self.neg_label_ = None

        self.updates_in_epoch_ = 0
        self.total_updates_ = 0

        self.zone0_ = 0
        self.zone1_ = 0
        self.zone2_ = 0
        self.zone3_ = 0

    @property
    def delta(self) -> float:
        return float(self.eps) * float(self.delta_ratio)

    def initialize(self, n_features: int, classes):
        if self.model_type == "classification":
            classes = np.asarray(classes)
            if len(classes) != 2:
                raise ValueError(f"C-system supports binary classification only, got classes={classes}")
            self.classes_ = classes
            self.neg_label_, self.pos_label_ = classes[0], classes[1]

        d = int(n_features) + (1 if self.fit_bias else 0)
        self._initialize_weights(d)

        self.updates_in_epoch_ = 0
        self.total_updates_ = 0
        self.constraint_checks_in_epoch_ = 0
        self.total_constraint_checks_ = 0
        self.zone0_ = self.zone1_ = self.zone2_ = self.zone3_ = 0
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.model_type == "classification":
            classes = np.unique(y)
            if len(classes) != 2:
                raise ValueError(f"C-system supports binary classification only, got classes={classes}")
        else:
            classes = None

        self.initialize(n_features=X.shape[1], classes=classes)
        
        for _ in range(self.epochs_count):
            self.partial_fit(X, y, classes=classes)

        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)

        if self.model_type == "classification" and self.classes_ is None:
            if classes is None:
                raise ValueError("For classification, provide `classes` on the first call.")
            self.initialize(n_features=X.shape[1], classes=classes)

        s = self._prepare_labels(y)

        if self.fit_bias:
            X = self._append_dummy_feature(X)

        if self.weights_ is None:
            self._initialize_weights(X.shape[1])

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

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.fit_bias:
            X = self._append_dummy_feature(X)
        return X.dot(self.weights_).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)

        if self.model_type == "classification":
            if self.pos_label_ is None or self.neg_label_ is None:
                raise RuntimeError("Class labels are not initialized.")
            return np.where(scores >= self.thresh, self.pos_label_, self.neg_label_)

        return scores

    def _initialize_weights(self, features_count: int):
        if self.init_mode == "zeros":
            self.weights_ = np.zeros(features_count, dtype=float)
        elif self.init_mode == "random":
            self.weights_ = self._rng.random(features_count) * self.init_scale
        else:
            raise ValueError("init_mode must be 'zeros' or 'random'")

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if self.model_type != "classification":
            return y.astype(float)

        if self.pos_label_ is None:
            raise RuntimeError("Call initialize(..., classes=...) or partial_fit(..., classes=...) first.")

        return np.where(y == self.pos_label_, 1.0, -1.0).astype(float)

    def _one_sample_step(self, x: np.ndarray, s: float):
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

    def _append_dummy_feature(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("_append_dummy_feature expects 2D array.")
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([X, ones])


import typing
import numpy as np
from sklearn.base import BaseEstimator


class Stripe(BaseEstimator):
    def __init__(
        self,
        mode: str,
        delta_m: float = 1e-10,
        epochs_count: int = 1,
        fit_bias: bool = True,
        lam: float = 0.0,
        regularize_bias: bool = False,
        model_type: str = "classification",
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.mode = mode
        self.epochs_count = int(epochs_count)
        self.lam = float(lam)
        self.regularize_bias = bool(regularize_bias)
        self.model_type = str(model_type)
        self.fit_bias = bool(fit_bias)

        self.shuffle = bool(shuffle)
        self.random_state = random_state

        self.delta_m = float(delta_m)
        self.thresh = 0.0

        self.kappa_hk = None
        self.a_hkm = None
        self.a_hm = None
        self.m = None

        self.classes_ = None
        self.neg_label_ = None
        self.pos_label_ = None

        self._rng = np.random.default_rng(self.random_state)

        # epoch-level counters
        self.updates_in_epoch_ = 0
        self.constraint_checks_in_epoch_ = 0
        self.active_samples_in_epoch_ = 0
        self.update_norm_sum_in_epoch_ = 0.0
        self.update_norm_max_in_epoch_ = 0.0

        # total counters
        self.total_updates_ = 0
        self.total_constraint_checks_ = 0
        self.total_active_samples_ = 0
        self.total_update_norm_sum_ = 0.0
        self.total_update_norm_max_ = 0.0

    # ============================================================
    # Counter helpers
    # ============================================================

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
    def update_rate_in_epoch_(self) -> float:
        if self.constraint_checks_in_epoch_ == 0:
            return np.nan
        return self.updates_in_epoch_ / self.constraint_checks_in_epoch_

    @property
    def active_sample_rate_in_epoch_(self) -> float:
        if not hasattr(self, "_last_epoch_samples_count") or self._last_epoch_samples_count == 0:
            return np.nan
        return self.active_samples_in_epoch_ / self._last_epoch_samples_count

    @property
    def mean_update_norm_in_epoch_(self) -> float:
        if self.updates_in_epoch_ == 0:
            return 0.0
        return self.update_norm_sum_in_epoch_ / self.updates_in_epoch_

    def get_epoch_counters(self) -> dict:
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

    # ============================================================
    # Initialization
    # ============================================================

    def initialize(self, n_features: int, classes=None):
        if self.model_type == "classification":
            if classes is None:
                raise ValueError("For classification, provide classes.")

            classes = np.asarray(classes)
            if len(classes) != 2:
                raise ValueError(
                    f"Stripe supports binary classification only, got classes={classes}"
                )

            self.classes_ = classes
            self.neg_label_, self.pos_label_ = classes[0], classes[1]

        features_count = int(n_features) + (1 if self.fit_bias else 0)

        self.kappa_hk = self._rng.random(features_count) * 0.01
        self.a_hkm = np.zeros((features_count, features_count), dtype=float)
        self.a_hm = np.zeros(features_count, dtype=float)
        self.m = -1

        self._reset_epoch_counters()
        self._reset_total_counters()

        return self

    # ============================================================
    # Fit / partial_fit
    # ============================================================

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.model_type == "classification":
            classes = np.unique(y)
            self.initialize(n_features=X.shape[1], classes=classes)
        else:
            self.initialize(n_features=X.shape[1], classes=None)

        s = self._prepare_labels(y)

        if self.fit_bias:
            X = self.append_dummy_feature(X)

        self.main_loop(X, s)
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)

        if self.kappa_hk is None:
            if self.model_type == "classification":
                if classes is None:
                    raise ValueError(
                        "For classification, provide `classes` on the first call to partial_fit()."
                    )
                self.initialize(n_features=X.shape[1], classes=classes)
            else:
                self.initialize(n_features=X.shape[1], classes=None)

        s = self._prepare_labels(y)

        if self.fit_bias:
            X = self.append_dummy_feature(X)

        n = X.shape[0]
        self._last_epoch_samples_count = n
        self._reset_epoch_counters()

        if self.shuffle and n > 1:
            idx = self._rng.permutation(n)
        else:
            idx = range(n)

        for i in idx:
            self.one_sample_step(X[i:i + 1], float(s[i]))

        return self

    def main_loop(self, X: np.ndarray, s: np.ndarray):
        self.m = -1
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
                self.m = -1

            for i in range(samples_count):
                self.one_sample_step(
                    Xp[i].reshape(1, features_size),
                    float(sp[i]),
                )

    # ============================================================
    # One sample update
    # ============================================================

    def one_sample_step(self, x: np.ndarray, s: float):
        features_count = x.shape[1]

        if self.mode == "single":
            self.m = 0
        else:
            self.m += 1

        m = self.m

        xxT = np.dot(
            x.reshape((features_count, 1)),
            x.reshape((1, features_count)),
        )

        if self.lam != 0.0:
            reg = self.lam * np.eye(features_count, dtype=float)
            if self.fit_bias and (not self.regularize_bias):
                reg[-1, -1] = 0.0
            xxT = xxT + reg

        self.a_hkm = (m / (m + 1)) * self.a_hkm + (1 / (m + 1)) * xxT
        self.a_hm = (m / (m + 1)) * self.a_hm + (1 / (m + 1)) * (
            s * x.reshape(features_count)
        )

        sample_had_update = False

        for h in self._rng.permutation(features_count).tolist():
            self.constraint_checks_in_epoch_ += 1
            self.total_constraint_checks_ += 1

            delta_hm = self.calculate_next_delta(h)

            if abs(delta_hm) >= self.delta_m:
                did_update = self.update_kappa(delta_hm, h)
                if did_update:
                    sample_had_update = True

        if sample_had_update:
            self.active_samples_in_epoch_ += 1
            self.total_active_samples_ += 1

    def calculate_next_delta(self, h: int) -> float:
        return float(np.dot(self.a_hkm[h], self.kappa_hk) - self.a_hm[h])

    def update_kappa(self, delta_hm: float, h: int) -> bool:
        denom = float(np.dot(self.a_hkm[h], self.a_hkm[h].T))

        if denom <= 0:
            return False

        coeff = delta_hm / denom
        update_vec = coeff * self.a_hkm[h]
        update_norm = float(np.linalg.norm(update_vec))

        self.kappa_hk = self.kappa_hk - update_vec

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

    # ============================================================
    # Prediction
    # ============================================================

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.kappa_hk is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.fit_bias:
            X = self.append_dummy_feature(X)

        return (X @ self.kappa_hk).reshape(-1)

    def predict(self, X: np.ndarray):
        scores = self.decision_function(X)

        if self.model_type == "classification":
            if self.pos_label_ is not None and self.neg_label_ is not None:
                return np.where(
                    scores >= self.thresh,
                    self.pos_label_,
                    self.neg_label_,
                )

            return np.array([self.sign(float(s)) for s in scores])

        return scores

    def sign(self, val: float) -> int:
        return 1 if val >= self.thresh else 0

    # ============================================================
    # Labels / bias
    # ============================================================

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)

        if self.model_type != "classification":
            return y.astype(float)

        if self.pos_label_ is None or self.neg_label_ is None:
            raise RuntimeError(
                "Call initialize(..., classes=...) or partial_fit(..., classes=...) first."
            )

        return np.where(y == self.pos_label_, 1.0, -1.0).astype(float)

    def append_dummy_feature(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "append_dummy_feature expects 2D array (n_samples, n_features)."
            )

        dummy_feature = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([X, dummy_feature])

