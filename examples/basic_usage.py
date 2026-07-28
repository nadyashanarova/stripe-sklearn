"""Basic usage of the four Stripe Learning estimators.

Run with:

    python examples/basic_usage.py

This script walks through:

1. Fitting each of the four estimators on a synthetic dataset.
2. Incremental learning with ``partial_fit``.
3. Using an estimator inside a scikit-learn ``Pipeline`` and
   ``GridSearchCV``.

It only requires the package itself (``pip install -e .``) plus
scikit-learn, which is already a dependency.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stripe_learning import (
    StripeCClassifier,
    StripeCKernelClassifier,
    StripeLClassifier,
    StripeLKernelClassifier,
)


def make_dataset():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=2,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)


def section(title):
    print(f"\n{'-' * len(title)}\n{title}\n{'-' * len(title)}")


def main():
    X_train_raw, X_test_raw, y_train, y_test = make_dataset()

    # Stripe algorithms use fixed absolute tolerances (eps, delta_m, ...),
    # so -- like most projection-based / margin-based methods -- they are
    # sensitive to feature scale. Scale features before fitting directly
    # (sections 1-2); section 3 shows the equivalent, CV-safe idiom of
    # putting the scaler inside a Pipeline instead (using the raw,
    # unscaled data).
    scaler = StandardScaler().fit(X_train_raw)
    X_train, X_test = scaler.transform(X_train_raw), scaler.transform(X_test_raw)

    # ------------------------------------------------------------------
    # 1. Fitting each of the four estimators
    # ------------------------------------------------------------------
    section("1. Fitting each estimator")

    estimators = {
        "StripeLClassifier (linear L-system)": StripeLClassifier(random_state=0),
        "StripeCClassifier (linear C-system)": StripeCClassifier(random_state=0),
        "StripeLKernelClassifier (kernel L-system)": StripeLKernelClassifier(
            random_state=0
        ),
        "StripeCKernelClassifier (kernel C-system)": StripeCKernelClassifier(
            random_state=0
        ),
    }

    for name, clf in estimators.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        scores = clf.decision_function(X_test[:3])
        print(f"{name}:")
        print(f"  test accuracy      = {acc:.3f}")
        print(f"  decision_function  = {np.round(scores, 3)}")
        print(f"  predict            = {clf.predict(X_test[:3])}")

    # ------------------------------------------------------------------
    # 2. Incremental learning with partial_fit
    # ------------------------------------------------------------------
    section("2. Incremental learning with partial_fit")

    clf = StripeLClassifier(random_state=0)
    classes = np.unique(y_train)
    batch_size = 50
    for start in range(0, len(X_train), batch_size):
        X_batch = X_train[start : start + batch_size]
        y_batch = y_train[start : start + batch_size]
        if start == 0:
            clf.partial_fit(X_batch, y_batch, classes=classes)
        else:
            clf.partial_fit(X_batch, y_batch)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"StripeLClassifier trained via {len(X_train) // batch_size} "
          f"partial_fit batches: test accuracy = {acc:.3f}")

    # Kernel estimators' partial_fit both accumulates the batch into the
    # kernel dictionary AND runs one Stripe correction epoch in the same
    # call -- see README.md for how this differs from the reference
    # implementation's two-step accumulate/epoch API.
    kernel_clf = StripeLKernelClassifier(random_state=0)
    for start in range(0, len(X_train), batch_size):
        X_batch = X_train[start : start + batch_size]
        y_batch = y_train[start : start + batch_size]
        if start == 0:
            kernel_clf.partial_fit(X_batch, y_batch, classes=classes)
        else:
            kernel_clf.partial_fit(X_batch, y_batch)

    acc = accuracy_score(y_test, kernel_clf.predict(X_test))
    print(f"StripeLKernelClassifier trained via partial_fit: "
          f"test accuracy = {acc:.3f}, dictionary size = {kernel_clf.m_}")

    # ------------------------------------------------------------------
    # 3. Pipeline + GridSearchCV
    # ------------------------------------------------------------------
    section("3. Pipeline + GridSearchCV")

    pipe = make_pipeline(StandardScaler(), StripeCClassifier(random_state=0))
    param_grid = {
        "stripecclassifier__eps": [0.05, 0.1, 0.2],
        "stripecclassifier__lam": [0.0, 0.05],
    }
    search = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="accuracy")
    search.fit(X_train_raw, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.3f}")
    print(
        "Held-out test accuracy: "
        f"{accuracy_score(y_test, search.predict(X_test_raw)):.3f}"
    )


if __name__ == "__main__":
    main()
